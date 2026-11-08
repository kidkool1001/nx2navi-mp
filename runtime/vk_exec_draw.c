// Draws, binds and dispatches of the Vulkan op-stream replay: the strong
// vk_exec_op_draw behind vk_exec.c's weak hook. The bind ops fill the VkExec slots; a draw opens the pass, resolves the
// bound program to translated shaders (the translator's answers come from the bound state), asks vk_pipeline.c for the
// pipeline, pushes (or allocates) set 0 as VkPipeInfo.bindings says and records the dynamic state, vertex and index
// buffers and the draw. A dispatch ends the pass and does the same for the compute shader. UPDATE_UB is a buffer copy
// from transient memory between full barriers.
//
// Per command buffer the replay keeps transient memory (support buffers, copied uniform ranges, generated indices),
// descriptor pools, image views and the storage-buffer ranges the GPU may write; all of it is freed, and the ranges are
// reported with vk_gpu_written, from one completion callback once the submission's fence is done (the write watch
// does not see GPU writes). Nothing here locks: it runs on the submitting thread under vk_sync_submit's lock.
//
// Unbound vertex streams read a zeroed buffer. The support buffer layout is the translator's SupportBuffer.
// Contains code Copyright (c) 2026 jhaizhou, Copyright (c) 2026 Peiikai (MIT; see THIRD_PARTY_NOTICES.md).
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "host.h"
#include "nvn_api.h"
#include "nvn_hmap.h"
#include "vk.h"

#ifndef NX2NAVI_TONEMAP_STANDIN
#define NX2NAVI_TONEMAP_STANDIN 1   // vk_tonemap.c, see untranslated_dispatch_fallback
#endif
#include "vk_exec.h"
#include "vk_gpa.h"
#include "platform/platform.h"
#include "vk_dlaa.h"
#include "multiplayer.h"

#define MAX_BINDINGS 128
#define MAX_DESC 1024
#define CHUNK_BYTES (4u << 20)
#define ZERO_BYTES 65536u
#define SUPPORT_BYTES 1456u   // the translator's SupportBuffer (std140): alpha_test, is_bgra[8], viewport_inverse, viewport_size, ...;
                              // 1408 pcss and 1424 shadow_spread (translator patches 0010, 0011), 1440 HDR (support_fill)
#define NVN_STAGE_VS 0
#define NVN_STAGE_FS 1
#define NVN_STAGE_CS 5
#define NVN_STAGE_TASK 6   // NVN2
#define NVN_STAGE_MESH 7
#define ONESHOT_WAIT_MS 30000u   // prepare_draw: how long a one-shot draw waits for its pipeline before it is skipped

enum { W_PROGRAM, W_SHADER, W_PIPELINE, W_NO_VS, W_INDEX, W_UB, W_BIND, W_SSBO_ALIGN, W_SSBO_SLOT, W_RESTART, W_INDIRECT, W_MESH, W_COUNT };

static void warn_once(int key, const char *fmt, ...) {
    static volatile int32_t seen[W_COUNT];
    if (__atomic_exchange_n(&seen[key], 1, __ATOMIC_RELAXED)) return;
    char msg[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    rt_log("vk: %s (logged once)", msg);
}

static void *grow(void *p, int *cap, size_t elem) {
    *cap = *cap ? *cap * 2 : 8;
    p = realloc(p, elem * (size_t)*cap);
    if (!p) rt_fatal("vk: out of memory");
    return p;
}

// ---- resources of one command buffer, freed by its completion callback

typedef struct { VkBuffer buf; VmaAllocation alloc; uint8_t *map; uint64_t size, used; } Chunk;
typedef struct {
    VkImage image;
    VkFormat format;
    VkImageViewType type;
    uint32_t aspect, level0, levels, layer0, layers, swz;
} ViewKey;
_Static_assert(sizeof(ViewKey) == 40, "ViewKey is hashed as bytes: no padding");
typedef struct { uint64_t addr, size; } Range;
typedef struct {
    Chunk *chunks;
    int nchunks, cchunks;
    VkDescriptorPool *pools;
    int npools, cpools;
    Range *written;
    int nwritten, cwritten;
} CbRes;

static CbRes *cur;
static uint64_t cur_fence;

// Descriptor pools and full-size transient chunks go back to these lists when their command buffer completes, instead
// of being destroyed and created again for the next one: each creation is a driver memory allocation (on Windows an
// allocation, a map and a paging-fence wait on the CPU), about ten of them per frame inside the replay.
#define SPARE_POOLS 64
#define SPARE_CHUNKS 16
static PlatMutex spare_lock = PLAT_MUTEX_INIT;
static VkDescriptorPool spare_pool[SPARE_POOLS];
static Chunk spare_chunk[SPARE_CHUNKS];
static int nspare_pool, nspare_chunk;

// State of the command buffer being recorded that the Vulkan command buffer itself holds.
static struct {
    VkPipeline pipe[2];                  // bound per bind point: 0 graphics, 1 compute
    VkPipeState st;                      // the state vk_pipeline_set_dynamic last recorded
    int st_valid;
    VkBuffer vb[17];                     // vertex buffers last bound
    VkDeviceSize vo[17], vsz[17], vstr[17];
    int vcls[17];                        // their pool class and pool record, for the census
    void *vpool[17];
    NVNbufferRange vin[16];              // the ranges they were resolved from, under vk_mem_gen vgen
    uint64_t vgen;
    int vb_valid;
    uint8_t support[SUPPORT_BYTES];      // last support buffer and where it lives
    float sup_in[3];                     // the support_fill arguments it was filled from
    VkBuffer sup_buf;
    VkDeviceSize sup_off;
    int sup_valid;
    uint8_t c0[3][VK_C0_BYTES];          // last c0 bank per stage (vertex, fragment, compute) and where it lives
    VkBuffer c0_buf[3];
    VkDeviceSize c0_off[3];
    int c0_valid[3];
    struct { p_bind_buf ssbo[16], ubo[16]; uint32_t dim[3], binst; uint8_t push[1024]; } c0_in[3];   // what vk_c0_fill read for c0[k]
    VkPipelineLayout ds_layout;          // graphics layout the sets below were last pushed or bound with
    VkDescriptorSet ds_bound[2];         // allocated set bound per set number (0: none or pushed since)
    VkDescriptorSet ds_last[2];          // last allocated and written set per set number, contents in DS
    VkDescriptorSetLayout ds_last_layout[2];
    int ds_pushed[2];                    // set number holds the push of DS.psig under ds_layout
    int res_valid;                       // the graphics sets bound hold what the inputs in RS give (bind_resources)
} B;

// Contents of B.ds_last: the set's packed descriptors (VkDescSlot), which with its layout determine every write. A draw
// whose allocated set would hold the same reuses the set instead of allocating and writing one (a per-draw
// cost). psig: the same for the last push per set number; a push equal to it under the same layout is not recorded
// again (the descriptors stay bound across pipeline binds of that layout).
#define DS_SIG_BYTES (MAX_DESC * sizeof(VkDescSlot))
static struct { uint32_t n[2], pn[2]; uint8_t sig[2][DS_SIG_BYTES], psig[2][DS_SIG_BYTES]; } DS;

// NX2NAVI_BIND_STATS counters (bind_stats).
static struct {
    int on;
    uint64_t draws, bind[2], desc[2], push, push_skip, set_new, set_reuse, ac_miss, same_prog, same_all, bind_skip;
} BS = {-1};

// The inputs of the last graphics bind_resources (bind_sig) and what its build counted for the traffic census, which a
// skipped build counts again. RS_BYTES bounds the bound-slot bytes a pipeline reads (RsList): per binding at most one
// c0 storage table, otherwise one bound range or handle per descriptor.
#define RS_BYTES (MAX_BINDINGS * sizeof(((VkExec *)0)->ssbo[0]) + MAX_DESC * sizeof(p_bind_buf))
typedef struct { int cls, use; uint64_t bytes; } RsCount;
static struct {
    int on;                              // NX2NAVI_BIND_SKIP: 1 skip (default), 2 check, 0 off
    uint32_t hn;                         // words of h the last signature filled; 0: none
    uint64_t h[24];                      // generations, feedback images, support inputs, base instance
    const VkPipeInfo *info;              // the pipeline whose RsList ranges body holds, as of bind op generation gen
    uint64_t gen;
    uint8_t body[RS_BYTES];
    int rec;                             // a graphics build is running: count_use records, scratch copies mark volatile
    int volatile_;                       // the build copied guest bytes into transient memory: a rebuild differs
    RsCount cnt[2 * VK_MCLS_N], pcnt[2 * VK_MCLS_N];
    int ncnt, pncnt, cnt_over;
    int cmds;                            // check mode: pushes, binds and set writes the build recorded
    uint32_t ckn;                        // check mode: the last build's packed descriptors (bytes in ck)
    uint8_t ck[DS_SIG_BYTES];
} RS = {-1};

// The bound-slot bytes of VkExec a pipeline's descriptors read, as ranges merged where they touch, built once per
// pipeline info: c0's storage table, the uniform and storage ranges of the bound slots and the texture and image
// handles. sup and c0: the support inputs and the base instance count too. p: storage buffers whose address the shader
// loads from a user uniform block; the address read from the block counts too. ok 0: never skipped (too many
// descriptors or such buffers).
#define RS_PTRS 32
typedef struct { uint32_t off, len; } RsRange;
typedef struct { int st; const GfxShaderRes *res; } RsPtr;
typedef struct { uint32_t n, bytes; uint8_t sup, c0, ok, np; RsPtr p[RS_PTRS]; RsRange r[]; } RsList;
static nvn_hmap rs_lists;                // VkPipeInfo pointer -> RsList, until shutdown
static const VkPipeInfo *rs_last;        // the last lookup
static const RsList *rs_last_l;

static uint32_t c0_dim[3], c0_binst;     // the dispatch size and base instance of the draw or dispatch being bound
static uint64_t bind_gen = 1;            // bumped by every bind op that writes ex->prog, ubo, ssbo, tex or img

static void cb_done(void *arg) {   // completion thread
    CbRes *r = arg;
    VkCtx *c = vk_ctx();
    for (int i = 0; i < r->nwritten; i++) vk_gpu_written(r->written[i].addr, r->written[i].size);
    for (int i = 0; i < r->npools; i++) {
        vkResetDescriptorPool(c->device, r->pools[i], 0);
        plat_mutex_lock(&spare_lock);
        int kept = nspare_pool < SPARE_POOLS;
        if (kept) spare_pool[nspare_pool++] = r->pools[i];
        plat_mutex_unlock(&spare_lock);
        if (!kept) vkDestroyDescriptorPool(c->device, r->pools[i], NULL);
    }
    for (int i = 0; i < r->nchunks; i++) {
        plat_mutex_lock(&spare_lock);
        int kept = r->chunks[i].size == CHUNK_BYTES && nspare_chunk < SPARE_CHUNKS;
        if (kept) spare_chunk[nspare_chunk++] = r->chunks[i];
        plat_mutex_unlock(&spare_lock);
        if (!kept) vmaDestroyBuffer(c->vma, r->chunks[i].buf, r->chunks[i].alloc);
    }
    free(r->written), free(r->pools), free(r->chunks), free(r);
}

void vk_exec_draw_new_cb(void) { memset(&B, 0, sizeof B); }

void vk_exec_draw_shaders(const GfxShader **vs, const GfxShader **fs) {
    *vs = B.st_valid ? B.st.vs : NULL;
    *fs = B.st_valid ? B.st.fs : NULL;
}

// Marks a dispatch about to be recorded (vk.h vk_pass_mark_begin); the caller ends it after the dispatch.
static VkPassMark *dispatch_mark(VkExec *ex, const GfxShader *cs, uint32_t x, uint32_t y, uint32_t z) {
    VkPassMark *m = vk_pass_mark_begin(ex->cb, VK_MARK_DISPATCH, ex->fence);
    if (!m) return NULL;
    m->step = ex->mark_dispatches++;
    m->w = x, m->h = y, m->d = z, m->draws = 0, m->ncolor = 0;
    m->first[0] = cs, m->first[1] = m->last[0] = m->last[1] = NULL;
    return m;
}

static CbRes *cb_res(VkExec *ex) {
    if (cur && cur_fence == ex->fence) return cur;
    cur = calloc(1, sizeof *cur);
    if (!cur) rt_fatal("vk: out of memory");
    cur_fence = ex->fence;
    memset(&B, 0, sizeof B);
    vk_completion_add(ex->fence, cb_done, cur);
    return cur;
}

static uint8_t *scratch(VkExec *ex, uint64_t size, VkBuffer *buf, VkDeviceSize *off) {
    CbRes *r = cb_res(ex);
    size = (size + 255) & ~255ull;
    Chunk *c = r->nchunks ? &r->chunks[r->nchunks - 1] : NULL;
    if (!c || c->used + size > c->size) {
        if (r->nchunks == r->cchunks) r->chunks = grow(r->chunks, &r->cchunks, sizeof *r->chunks);
        c = &r->chunks[r->nchunks];
        uint64_t n = size > CHUNK_BYTES ? size : CHUNK_BYTES;
        int reused = 0;
        if (n == CHUNK_BYTES) {
            plat_mutex_lock(&spare_lock);
            if (nspare_chunk) *c = spare_chunk[--nspare_chunk], reused = 1;
            plat_mutex_unlock(&spare_lock);
        }
        if (reused) {
            c->used = 0;
            r->nchunks++;
            *buf = c->buf;
            *off = 0;
            c->used = size;
            return c->map;
        }
        VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = n,
                                 .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                          VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                          VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
        // HOST_COHERENT: the CPU fills these chunks and never flushes them. Intel's integrated GPUs (ANV, Meteor Lake)
        // offer host-cached memory only without coherence, which VMA picks for random host access: the GPU then read
        // what an earlier submission left in the chunk (an UpdateUniformBuffer payload of another draw: a lighting
        // shader's step count of 0x3f800000 instead of 128, and a loop of a billion steps).
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                      .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
                                      .requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT};
        VmaAllocationInfo info;
        VkResult res = vmaCreateBuffer(vk_ctx()->vma, &bi, &ai, &c->buf, &c->alloc, &info);
        if (res != VK_SUCCESS) rt_fatal("vk: transient buffer of %llu bytes failed (%d)", (unsigned long long)n, (int)res);
        c->map = info.pMappedData, c->size = n, c->used = 0;
        r->nchunks++;
    }
    *buf = c->buf;
    *off = c->used;
    uint8_t *p = c->map + c->used;
    c->used += size;
    return p;
}

static void note_written(VkExec *ex, uint64_t addr, uint64_t size) {
    CbRes *r = cb_res(ex);
    for (int i = 0; i < r->nwritten; i++)
        if (r->written[i].addr == addr && r->written[i].size == size) return;
    if (r->nwritten == r->cwritten) r->written = grow(r->written, &r->cwritten, sizeof *r->written);
    r->written[r->nwritten++] = (Range){addr, size};
    vk_mem_gpu_pending(addr, size);
}

// ---- stand-ins for what a binding names but the game did not bind (zeros, a 1x1 image, the default sampler)

enum { DV_1D, DV_1D_ARRAY, DV_2D, DV_2D_ARRAY, DV_3D, DV_CUBE, DV_CUBE_ARRAY, DV_2D_MS, DV_2D_MS_ARRAY, DV_COUNT };

static struct {
    int ready;
    VkBuffer zbuf;
    VmaAllocation zalloc;
    VkBufferView zview;
    VkBuffer sink;                       // ssbo_info: an unresolved storage buffer a shader writes (zbuf stays zero)
    VmaAllocation sink_alloc;
    VkSampler samp, samp_shadow;
    struct { VkImage image; VmaAllocation alloc; VkImageView view, sview; } img[DV_COUNT];
} Z;

static const struct { VkImageType t; VkImageViewType vt; uint32_t layers; VkSampleCountFlagBits s; VkImageCreateFlags f; } dinfo[DV_COUNT] = {
    {VK_IMAGE_TYPE_1D, VK_IMAGE_VIEW_TYPE_1D, 1, VK_SAMPLE_COUNT_1_BIT, 0},
    {VK_IMAGE_TYPE_1D, VK_IMAGE_VIEW_TYPE_1D_ARRAY, 1, VK_SAMPLE_COUNT_1_BIT, 0},
    {VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D, 1, VK_SAMPLE_COUNT_1_BIT, 0},
    {VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D_ARRAY, 1, VK_SAMPLE_COUNT_1_BIT, 0},
    {VK_IMAGE_TYPE_3D, VK_IMAGE_VIEW_TYPE_3D, 1, VK_SAMPLE_COUNT_1_BIT, 0},
    {VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_CUBE, 6, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT},
    {VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_CUBE_ARRAY, 6, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT},
    {VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D, 1, VK_SAMPLE_COUNT_4_BIT, 0},
    {VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D_ARRAY, 1, VK_SAMPLE_COUNT_4_BIT, 0}};

// A 1x1 stand-in image of view type dinfo[i] in `fmt`, moved to GENERAL by the upload batch (which reaches the queue before
// the submission that uses it). *sview is the view a storage descriptor takes (a cube becomes a 2D array).
static void make_dummy(int i, VkFormat fmt, VkImage *image, VmaAllocation *alloc, VkImageView *view, VkImageView *sview) {
    VkCtx *c = vk_ctx();
    int ms = dinfo[i].s != VK_SAMPLE_COUNT_1_BIT;
    VkImageCreateInfo ii = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .flags = dinfo[i].f, .imageType = dinfo[i].t,
                            .format = fmt, .extent = {1, 1, 1}, .mipLevels = 1, .arrayLayers = dinfo[i].layers,
                            .samples = dinfo[i].s, .tiling = VK_IMAGE_TILING_OPTIMAL,
                            .usage = VK_IMAGE_USAGE_SAMPLED_BIT | (ms ? 0 : VK_IMAGE_USAGE_STORAGE_BIT),
                            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    VmaAllocationCreateInfo ia = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    if (vmaCreateImage(c->vma, &ii, &ia, image, alloc, NULL) != VK_SUCCESS) rt_fatal("vk: stand-in image %d failed", i);
    VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = *image,
                                .viewType = dinfo[i].vt, .format = fmt,
                                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, dinfo[i].layers}};
    if (vkCreateImageView(c->device, &vi, NULL, view) != VK_SUCCESS) rt_fatal("vk: stand-in view failed");
    *sview = *view;
    if (!ms && (dinfo[i].vt == VK_IMAGE_VIEW_TYPE_CUBE || dinfo[i].vt == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY)) {
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        if (vkCreateImageView(c->device, &vi, NULL, sview) != VK_SUCCESS) rt_fatal("vk: stand-in view failed");
    }
    VkImageMemoryBarrier2 bar = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                 .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                 .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                                 .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                 .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                 .image = *image, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, dinfo[i].layers}};
    VkCommandBuffer cb = vk_upload_begin();
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &bar};
    vkCmdPipelineBarrier2(cb, &di);
    vk_upload_end();
}

// Storage image stand-ins in the format a shader declares (GfxShaderRes.format: the translator's TextureFormat order); Vulkan wants
// the view format to match the declaration. Created on first use.
static const VkFormat tex_format[40] = {
    VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8_UNORM, VK_FORMAT_R8_SNORM, VK_FORMAT_R8_UINT, VK_FORMAT_R8_SINT,
    VK_FORMAT_R16_SFLOAT, VK_FORMAT_R16_UNORM, VK_FORMAT_R16_SNORM, VK_FORMAT_R16_UINT, VK_FORMAT_R16_SINT,
    VK_FORMAT_R32_SFLOAT, VK_FORMAT_R32_UINT, VK_FORMAT_R32_SINT, VK_FORMAT_R8G8_UNORM, VK_FORMAT_R8G8_SNORM,
    VK_FORMAT_R8G8_UINT, VK_FORMAT_R8G8_SINT, VK_FORMAT_R16G16_SFLOAT, VK_FORMAT_R16G16_UNORM, VK_FORMAT_R16G16_SNORM,
    VK_FORMAT_R16G16_UINT, VK_FORMAT_R16G16_SINT, VK_FORMAT_R32G32_SFLOAT, VK_FORMAT_R32G32_UINT, VK_FORMAT_R32G32_SINT,
    VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_SNORM, VK_FORMAT_R8G8B8A8_UINT, VK_FORMAT_R8G8B8A8_SINT,
    VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16B16A16_UNORM, VK_FORMAT_R16G16B16A16_SNORM, VK_FORMAT_R16G16B16A16_UINT,
    VK_FORMAT_R16G16B16A16_SINT, VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_R32G32B32A32_UINT, VK_FORMAT_R32G32B32A32_SINT,
    VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_FORMAT_A2B10G10R10_UINT_PACK32, VK_FORMAT_B10G11R11_UFLOAT_PACK32};
static struct { int used, di; VkFormat fmt; VkImage image; VmaAllocation alloc; VkImageView view, sview; } SD[48];

static VkImageView storage_dummy(int di, int code) {
    VkFormat fmt = code > 0 && code < 40 ? tex_format[code] : VK_FORMAT_R8G8B8A8_UNORM;
    if (fmt == VK_FORMAT_R8G8B8A8_UNORM || dinfo[di].s != VK_SAMPLE_COUNT_1_BIT) return Z.img[di].sview;
    for (int i = 0; i < 48; i++) {
        if (!SD[i].used) {
            SD[i].used = 1, SD[i].di = di, SD[i].fmt = fmt;
            make_dummy(di, fmt, &SD[i].image, &SD[i].alloc, &SD[i].view, &SD[i].sview);
            return SD[i].sview;
        }
        if (SD[i].di == di && SD[i].fmt == fmt) return SD[i].sview;
    }
    return Z.img[di].sview;
}

static void dummies_init(void) {
    if (Z.ready) return;
    VkCtx *c = vk_ctx();
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = ZERO_BYTES,
                             .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                      VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                  .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
                                  .requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT};   // the memset below is not flushed
    VmaAllocationInfo info;
    if (vmaCreateBuffer(c->vma, &bi, &ai, &Z.zbuf, &Z.zalloc, &info) != VK_SUCCESS) rt_fatal("vk: zero buffer failed");
    memset(info.pMappedData, 0, ZERO_BYTES);
    VkBufferViewCreateInfo bv = {.sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO, .buffer = Z.zbuf,
                                 .format = VK_FORMAT_R32G32B32A32_SFLOAT, .range = ZERO_BYTES};
    if (vkCreateBufferView(c->device, &bv, NULL, &Z.zview) != VK_SUCCESS) rt_fatal("vk: zero buffer view failed");
    VkBufferCreateInfo sbi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = ZERO_BYTES,
                              .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VmaAllocationCreateInfo sai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    if (vmaCreateBuffer(c->vma, &sbi, &sai, &Z.sink, &Z.sink_alloc, NULL) != VK_SUCCESS) rt_fatal("vk: sink buffer failed");
    samp_par sp;
    memset(&sp, 0, sizeof sp);
    sp.lod_max = 1000.0f;
    Z.samp = vk_sampler_for(&sp);
    sp.cmp_mode = 1, sp.cmp_func = NVN_DEPTH_FUNC_ALWAYS;
    Z.samp_shadow = vk_sampler_for(&sp);

    for (int i = 0; i < DV_COUNT; i++)
        make_dummy(i, VK_FORMAT_R8G8B8A8_UNORM, &Z.img[i].image, &Z.img[i].alloc, &Z.img[i].view, &Z.img[i].sview);
    Z.ready = 1;
}

static void twins_release(VkCtx *c);

// Releases the stand-ins and the depth twins. The device must be idle; the game path leaves them to process exit like
// the pipelines.
void vk_exec_draw_shutdown(void) {
    if (!Z.ready) return;
    VkCtx *c = vk_ctx();
    twins_release(c);
    for (int i = 0; i < DV_COUNT; i++) {
        if (Z.img[i].sview != Z.img[i].view) vkDestroyImageView(c->device, Z.img[i].sview, NULL);
        vkDestroyImageView(c->device, Z.img[i].view, NULL);
        vmaDestroyImage(c->vma, Z.img[i].image, Z.img[i].alloc);
    }
    for (int i = 0; i < 48 && SD[i].used; i++) {
        if (SD[i].sview != SD[i].view) vkDestroyImageView(c->device, SD[i].sview, NULL);
        vkDestroyImageView(c->device, SD[i].view, NULL);
        vmaDestroyImage(c->vma, SD[i].image, SD[i].alloc);
    }
    memset(SD, 0, sizeof SD);
    nvn_hm_clear(&rs_lists, free);
    rs_last = NULL, rs_last_l = NULL;
    for (int i = 0; i < nspare_pool; i++) vkDestroyDescriptorPool(c->device, spare_pool[i], NULL);
    for (int i = 0; i < nspare_chunk; i++) vmaDestroyBuffer(c->vma, spare_chunk[i].buf, spare_chunk[i].alloc);
    nspare_pool = nspare_chunk = 0;
    vk_usnap_shutdown();
    vkDestroyBufferView(c->device, Z.zview, NULL);
    vmaDestroyBuffer(c->vma, Z.zbuf, Z.zalloc);
    if (Z.sink) vmaDestroyBuffer(c->vma, Z.sink, Z.sink_alloc);
    memset(&Z, 0, sizeof Z);
}

// ---- resource types

static VkImageViewType res_view_type(uint16_t t) {
    int arr = t & 0x100;
    switch (t & 0xff) {
    case 1: return arr ? VK_IMAGE_VIEW_TYPE_1D_ARRAY : VK_IMAGE_VIEW_TYPE_1D;
    case 4: return VK_IMAGE_VIEW_TYPE_3D;
    case 5: return arr ? VK_IMAGE_VIEW_TYPE_CUBE_ARRAY : VK_IMAGE_VIEW_TYPE_CUBE;
    default: return arr ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
    }
}

static int dummy_index(VkImageViewType vt, int ms) {
    switch (vt) {
    case VK_IMAGE_VIEW_TYPE_1D: return DV_1D;
    case VK_IMAGE_VIEW_TYPE_1D_ARRAY: return DV_1D_ARRAY;
    case VK_IMAGE_VIEW_TYPE_3D: return DV_3D;
    case VK_IMAGE_VIEW_TYPE_CUBE: return DV_CUBE;
    case VK_IMAGE_VIEW_TYPE_CUBE_ARRAY: return DV_CUBE_ARRAY;
    case VK_IMAGE_VIEW_TYPE_2D_ARRAY: return ms ? DV_2D_MS_ARRAY : DV_2D_ARRAY;
    default: return ms ? DV_2D_MS : DV_2D;
    }
}

// GFX_TEX_* of a bound texture slot (the translator's tex_type / img_type answers).
static int gfx_type_of(const VkTexImage *img, const view_t *v, int hv) {
    if (hv && v->set_target) return vk_pipeline_tex_type(v->target);
    int ms = img->samples > 1;
    switch (img->view_type) {
    case VK_IMAGE_VIEW_TYPE_1D: return GFX_TEX_1D;
    case VK_IMAGE_VIEW_TYPE_1D_ARRAY: return GFX_TEX_1D_ARRAY;
    case VK_IMAGE_VIEW_TYPE_2D_ARRAY: return ms ? GFX_TEX_2D_MS_ARRAY : GFX_TEX_2D_ARRAY;
    case VK_IMAGE_VIEW_TYPE_3D: return GFX_TEX_3D;
    case VK_IMAGE_VIEW_TYPE_CUBE: return GFX_TEX_CUBE;
    case VK_IMAGE_VIEW_TYPE_CUBE_ARRAY: return GFX_TEX_CUBE_ARRAY;
    default: return ms ? GFX_TEX_2D_MS : GFX_TEX_2D;
    }
}

static VkComponentSwizzle swz_of(uint8_t s) {
    switch (s) {
    case NVN_TEXTURE_SWIZZLE_ZERO: return VK_COMPONENT_SWIZZLE_ZERO;
    case NVN_TEXTURE_SWIZZLE_ONE: return VK_COMPONENT_SWIZZLE_ONE;
    case NVN_TEXTURE_SWIZZLE_R: return VK_COMPONENT_SWIZZLE_R;
    case NVN_TEXTURE_SWIZZLE_G: return VK_COMPONENT_SWIZZLE_G;
    case NVN_TEXTURE_SWIZZLE_B: return VK_COMPONENT_SWIZZLE_B;
    default: return VK_COMPONENT_SWIZZLE_A;
    }
}

// A view in STENCIL mode samples the stencil plane (TextureViewSetDepthStencilMode); otherwise depth wins. The
// translator declares every sampled image float and the shader takes the raw bits, as for integer formats.
VkImageAspectFlags vk_exec_view_aspect(const VkTexImage *img, const view_t *v, int hv) {
    if (hv && v->set_dsmode && v->dsmode == NVN_TEXTURE_DEPTH_STENCIL_MODE_STENCIL && (img->aspect & VK_IMAGE_ASPECT_STENCIL_BIT))
        return VK_IMAGE_ASPECT_STENCIL_BIT;
    return img->aspect & VK_IMAGE_ASPECT_DEPTH_BIT     ? VK_IMAGE_ASPECT_DEPTH_BIT
           : img->aspect & VK_IMAGE_ASPECT_STENCIL_BIT ? VK_IMAGE_ASPECT_STENCIL_BIT
                                                       : VK_IMAGE_ASPECT_COLOR_BIT;
}

// Image views of registered textures, kept for their image's lifetime: made per command buffer they were created again
// every frame (vkCreateImageView took about 8% of the fight replay on the 7900X). Used under the texture lock, as
// vk_exec_image_gone is (destroy_tex), which hands the views of a destroyed image to the deferred release with it.
typedef struct ViewNode { struct ViewNode *next; ViewKey k; VkImageView view; } ViewNode;
static nvn_hmap view_of;                 // ViewKey -> VkImageView
static nvn_hmap views_by_image;          // VkImage -> its first ViewNode

// Integer formats a texture view may give an image (nvnTextureViewSetFormat): the Vulkan format and the translator's
// TextureFormat code (gfx_shader.c fmt_names) the translator declares a storage image with. A shader that packs texels
// itself writes raw words through such a view (Ocarina's last post-process pass stores R11G11B10 bits through an R32UI
// view of an R11G11B10F target); through a view of the texture's own format the store converted them (a black picture
// with red blobs). 0: not an integer format.
static int int_view_format(uint32_t nvn, VkFormat *vk) {
    static const struct { uint16_t nvn; uint8_t code; VkFormat vk; } T[] = {
        {NVN_FORMAT_R8UI, 3, VK_FORMAT_R8_UINT}, {NVN_FORMAT_R8I, 4, VK_FORMAT_R8_SINT},
        {NVN_FORMAT_R16UI, 8, VK_FORMAT_R16_UINT}, {NVN_FORMAT_R16I, 9, VK_FORMAT_R16_SINT},
        {NVN_FORMAT_R32UI, 11, VK_FORMAT_R32_UINT}, {NVN_FORMAT_R32I, 12, VK_FORMAT_R32_SINT},
        {NVN_FORMAT_RG8UI, 15, VK_FORMAT_R8G8_UINT}, {NVN_FORMAT_RG8I, 16, VK_FORMAT_R8G8_SINT},
        {NVN_FORMAT_RG16UI, 20, VK_FORMAT_R16G16_UINT}, {NVN_FORMAT_RG16I, 21, VK_FORMAT_R16G16_SINT},
        {NVN_FORMAT_RG32UI, 23, VK_FORMAT_R32G32_UINT}, {NVN_FORMAT_RG32I, 24, VK_FORMAT_R32G32_SINT},
        {NVN_FORMAT_RGBA8UI, 27, VK_FORMAT_R8G8B8A8_UINT}, {NVN_FORMAT_RGBA8I, 28, VK_FORMAT_R8G8B8A8_SINT},
        {NVN_FORMAT_RGBA16UI, 32, VK_FORMAT_R16G16B16A16_UINT}, {NVN_FORMAT_RGBA16I, 33, VK_FORMAT_R16G16B16A16_SINT},
        {NVN_FORMAT_RGBA32UI, 35, VK_FORMAT_R32G32B32A32_UINT}, {NVN_FORMAT_RGBA32I, 36, VK_FORMAT_R32G32B32A32_SINT},
        {NVN_FORMAT_RGB10A2UI, 38, VK_FORMAT_A2B10G10R10_UINT_PACK32}};
    for (size_t i = 0; i < sizeof T / sizeof *T; i++)
        if (T[i].nvn == nvn) {
            if (vk) *vk = T[i].vk;
            return T[i].code;
        }
    return 0;
}

// Bytes per texel of the formats a storage view may reinterpret (int_view_format's and the 32/64/128-bit float and
// unorm ones they alias); 0 unknown.
static int texel_bytes(VkFormat f) {
    switch (f) {
    case VK_FORMAT_R8_UINT: case VK_FORMAT_R8_SINT: case VK_FORMAT_R8_UNORM: return 1;
    case VK_FORMAT_R16_UINT: case VK_FORMAT_R16_SINT: case VK_FORMAT_R16_SFLOAT: case VK_FORMAT_R16_UNORM:
    case VK_FORMAT_R8G8_UINT: case VK_FORMAT_R8G8_SINT: case VK_FORMAT_R8G8_UNORM: return 2;
    case VK_FORMAT_R32_UINT: case VK_FORMAT_R32_SINT: case VK_FORMAT_R32_SFLOAT: case VK_FORMAT_R16G16_UINT:
    case VK_FORMAT_R16G16_SINT: case VK_FORMAT_R16G16_SFLOAT: case VK_FORMAT_R16G16_UNORM: case VK_FORMAT_R8G8B8A8_UINT:
    case VK_FORMAT_R8G8B8A8_SINT: case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_B8G8R8A8_UNORM:
    case VK_FORMAT_A2B10G10R10_UINT_PACK32: case VK_FORMAT_A2B10G10R10_UNORM_PACK32: case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
        return 4;
    case VK_FORMAT_R32G32_UINT: case VK_FORMAT_R32G32_SINT: case VK_FORMAT_R32G32_SFLOAT: case VK_FORMAT_R16G16B16A16_UINT:
    case VK_FORMAT_R16G16B16A16_SINT: case VK_FORMAT_R16G16B16A16_SFLOAT: case VK_FORMAT_R16G16B16A16_UNORM: return 8;
    case VK_FORMAT_R32G32B32A32_UINT: case VK_FORMAT_R32G32B32A32_SINT: case VK_FORMAT_R32G32B32A32_SFLOAT: return 16;
    default: return 0;
    }
}

// R11G11B10F images written this frame as raw words (storage_view_format chose an integer view): Ocarina's last
// post-process pass stores shared-exponent RGB9E5 texels into its R11G11B10F scene target that way, and the upscaler
// that reads them (tensor-core, untranslatable) decodes them itself; the fallback blit must too (raw_9e5).
static struct { VkImage img; uint64_t frame; } RAW[8];

static void raw_note(VkImage img) {
    uint64_t f = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    int k = 0;
    for (int i = 0; i < 8; i++) {
        if (RAW[i].img == img) { RAW[i].frame = f; return; }
        if (RAW[i].frame < RAW[k].frame) k = i;
    }
    RAW[k].img = img, RAW[k].frame = f;
}

static int raw_9e5(VkImage img, VkFormat f) {
    if (f != VK_FORMAT_B10G11R11_UFLOAT_PACK32) return 0;
    uint64_t now = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    for (int i = 0; i < 8; i++)
        if (RAW[i].img == img && now - RAW[i].frame <= 1) return 1;
    return 0;
}

// vk_dump.c: 1 when img was written as raw RGB9E5 words this frame (dump it decoded as such).
int vk_exec_raw_9e5(VkImage img, VkFormat f) { return raw_9e5(img, f); }

// The format a storage view of img takes: the view's integer format when it sets one of the image's texel size (the
// image is MUTABLE_FORMAT, vk_texture.c), else the image's own.
static VkFormat storage_view_format(const VkTexImage *img, const view_t *v, int hv) {
    VkFormat f;
    if (hv && v->set_format && int_view_format(v->format, &f) && texel_bytes(f) && texel_bytes(f) == texel_bytes(img->format)) {
        if (img->format == VK_FORMAT_B10G11R11_UFLOAT_PACK32) raw_note(img->image);
        return f;
    }
    return img->format;
}

// An image view of a registered texture for a shader resource of type `rtype` (GfxShaderRes.type); VK_NULL_HANDLE when
// the image cannot serve it (the caller binds a stand-in). Texture lock held.
static VkImageView get_view(VkExec *ex, const VkTexImage *img, const view_t *v, int hv, uint16_t rtype, int storage) {
    VkImageViewType vt = res_view_type(rtype);
    int ms = (rtype & 0x200) != 0;
    if ((img->samples > 1) != ms || (storage && ms)) return VK_NULL_HANDLE;
    if (storage && !(img->usage & VK_IMAGE_USAGE_STORAGE_BIT)) return VK_NULL_HANDLE;
    int cube = vt == VK_IMAGE_VIEW_TYPE_CUBE || vt == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY;
    int ok = 0;
    switch (vt) {
    case VK_IMAGE_VIEW_TYPE_1D:
    case VK_IMAGE_VIEW_TYPE_1D_ARRAY: ok = img->type == VK_IMAGE_TYPE_1D; break;
    case VK_IMAGE_VIEW_TYPE_3D: ok = img->type == VK_IMAGE_TYPE_3D; break;
    case VK_IMAGE_VIEW_TYPE_CUBE:
    case VK_IMAGE_VIEW_TYPE_CUBE_ARRAY:
        ok = img->type == VK_IMAGE_TYPE_2D && (img->view_type == VK_IMAGE_VIEW_TYPE_CUBE || img->view_type == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY);
        break;
    default: ok = img->type == VK_IMAGE_TYPE_2D; break;
    }
    if (!ok) return VK_NULL_HANDLE;
    uint32_t l0 = hv && v->set_levels ? v->base_level : 0;
    if (l0 >= img->levels) return VK_NULL_HANDLE;
    uint32_t nl = hv && v->set_levels && v->nlevels ? v->nlevels : img->levels - l0;
    if (l0 + nl > img->levels) nl = img->levels - l0;
    if (storage) nl = 1;
    uint32_t y0 = 0, ny = 1;
    if (vt != VK_IMAGE_VIEW_TYPE_3D) {
        y0 = hv && v->set_layers ? v->min_layer : 0;
        if (y0 >= img->layers) return VK_NULL_HANDLE;
        uint32_t rest = hv && v->set_layers && v->nlayers ? v->nlayers : img->layers - y0;
        if (y0 + rest > img->layers) rest = img->layers - y0;
        if (vt == VK_IMAGE_VIEW_TYPE_1D || vt == VK_IMAGE_VIEW_TYPE_2D) ny = 1;
        else if (vt == VK_IMAGE_VIEW_TYPE_CUBE) ny = 6;
        else if (vt == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY) ny = rest / 6 * 6;
        else ny = rest;
        if (!ny || y0 + ny > img->layers) return VK_NULL_HANDLE;
    }
    if (storage && cube) vt = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    uint32_t aspect = vk_exec_view_aspect(img, v, hv);
    uint8_t sw[4] = {NVN_TEXTURE_SWIZZLE_R, NVN_TEXTURE_SWIZZLE_G, NVN_TEXTURE_SWIZZLE_B, NVN_TEXTURE_SWIZZLE_A};
    if (!storage) {   // view selectors replace the texture's, then the format remap
        memcpy(sw, hv && v->set_swizzle ? v->swz : img->swz, 4);
        if (img->alpha_one || img->chan[0])
            for (int i = 0; i < 4; i++) {
                if (img->alpha_one && sw[i] == NVN_TEXTURE_SWIZZLE_A) sw[i] = NVN_TEXTURE_SWIZZLE_ONE;
                else if (img->chan[0] && sw[i] >= NVN_TEXTURE_SWIZZLE_R && sw[i] <= NVN_TEXTURE_SWIZZLE_A)
                    sw[i] = img->chan[sw[i] - NVN_TEXTURE_SWIZZLE_R];
            }
    }
    uint32_t swz = (uint32_t)sw[0] | (uint32_t)sw[1] << 8 | (uint32_t)sw[2] << 16 | (uint32_t)sw[3] << 24;
    VkFormat vf = storage ? storage_view_format(img, v, hv) : img->format;
    ViewKey k = {img->image, vf, vt, aspect, l0, nl, y0, ny, swz};
    if (!view_of.klen) nvn_hm_init(&view_of, sizeof(ViewKey)), nvn_hm_init(&views_by_image, sizeof(VkImage));
    VkImageView found = (VkImageView)nvn_hm_get(&view_of, &k);
    if (found) return found;
    VkImageViewCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = img->image, .viewType = vt,
                                .format = vf,
                                .components = {swz_of(sw[0]), swz_of(sw[1]), swz_of(sw[2]), swz_of(sw[3])},
                                .subresourceRange = {aspect, l0, nl, y0, ny}};
    VkImageViewUsageCreateInfo use = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
                                      .usage = VK_IMAGE_USAGE_SAMPLED_BIT};
    if (img->block_views) ci.pNext = &use;   // the image's COLOR_ATTACHMENT usage is its block views' (vk_texture.c)
    VkImageView view;
    if (vkCreateImageView(vk_ctx()->device, &ci, NULL, &view) != VK_SUCCESS) return VK_NULL_HANDLE;
    ViewNode *n = malloc(sizeof *n);
    if (!n) rt_fatal("vk: out of memory");
    *n = (ViewNode){nvn_hm_get(&views_by_image, &img->image), k, view};
    nvn_hm_put(&views_by_image, &img->image, n);
    nvn_hm_put(&view_of, &k, (void *)view);
    return view;
}

// An attachment view of a registered texture, kept with its sampled views for the image's lifetime: made per command
// buffer, every render target's views were created again each frame (vkCreateImageView under vk_exec_pass_begin, 2%
// of the fight replay on the 7900X). Attachment keys carry a swizzle word no sampled key has (those hold four NVN
// selectors, each below 8). The texture lock guards the tables, as for get_view; vk_exec_pass_begin may already hold it.
VkImageView vk_exec_target_view(const VkTexImage *img, VkFormat format, VkImageViewType type, uint32_t aspect,
                                uint32_t level, uint32_t base, uint32_t layers, int blocks) {
    ViewKey k = {img->image, format, type, aspect, level, 1, base, layers, blocks ? 0xfffffffeu : 0xffffffffu};
    vk_texture_lock();
    if (!view_of.klen) nvn_hm_init(&view_of, sizeof(ViewKey)), nvn_hm_init(&views_by_image, sizeof(VkImage));
    VkImageView view = (VkImageView)nvn_hm_get(&view_of, &k);
    if (!view) {
        VkImageViewCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = img->image,
                                    .viewType = type, .format = format, .subresourceRange = {aspect, level, 1, base, layers}};
        VkImageViewUsageCreateInfo use = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
                                          .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT};
        if (blocks) ci.pNext = &use;   // not the BC image's sampling and transfer usages
        if (vkCreateImageView(vk_ctx()->device, &ci, NULL, &view) != VK_SUCCESS)
            rt_fatal("vk: render target view (format %d) could not be created", (int)format);
        ViewNode *n = malloc(sizeof *n);
        if (!n) rt_fatal("vk: out of memory");
        *n = (ViewNode){nvn_hm_get(&views_by_image, &img->image), k, view};
        nvn_hm_put(&views_by_image, &img->image, n);
        nvn_hm_put(&view_of, &k, (void *)view);
    }
    vk_texture_unlock();
    return view;
}

void vk_exec_image_gone(VkImage image) {
    vk_exec_bc_gone(image);
    vk_hdr_image_gone(image);
    if (!views_by_image.klen) return;
    for (ViewNode *n = nvn_hm_del(&views_by_image, &image), *next; n; n = next) {
        next = n->next;
        nvn_hm_del(&view_of, &n->k);
        vk_defer_view(n->view);
        free(n);
    }
}

// ---- program to shaders

// (program, stage, answers) -> shader, 4-way set-associative, most recently used first. A miss costs a SHA-256 of the
// whole shader code (gfx_shader.c req_init), slow on CPUs without SHA instructions: a field frame looks up ~3000 and
// the 2048-slot direct-mapped table this replaced missed 3% of them on conflicts.
#define SHADER_SETS 2048
#define SHADER_WAYS 4
typedef struct { uint64_t serial; int stage; int used; GfxShaderAnswers a; const GfxShader *s; } ShaderEnt;
static ShaderEnt shader_tab[SHADER_SETS][SHADER_WAYS];

// Stage and code length of a control section. NVN1 (0x880 bytes): stage at +0x714, length at +0x704, c1 offset at
// +0x700, workgroup size at +0x748. NVN2 (Switch 2, 0xe0 bytes, magic 0x98761234 at +0): stage in bits 16+ of +0x54,
// length at +0x38, workgroup size at +0x58, no c1 part (gfx_shader.c, "NVN2").
static int ctl_nvn2(const uint8_t *c) {
    uint32_t magic;
    memcpy(&magic, c, 4);
    return magic == 0x98761234u;
}
static void ctl_stage_len(const uint8_t *c, uint32_t *st, uint32_t *len) {
    if (ctl_nvn2(c)) {
        uint32_t c1s, c1e;
        memcpy(st, c + 0x54, 4), *st >>= 16;
        memcpy(len, c + 0x38, 4);
        // NVN2: the c1 constants (BRX jump tables) follow the code from +0x40 to +0x44, in the same storage; the
        // length covers them so gfx_shader.c req_init sends them (with the code length alone it never could)
        memcpy(&c1s, c + 0x40, 4), memcpy(&c1e, c + 0x44, 4);
        if (c1s >= *len && c1e > c1s && c1e - *len < (1u << 20)) *len = c1e;
    } else {
        memcpy(st, c + 0x714, 4);
        memcpy(len, c + 0x704, 4);
    }
}
static uint32_t ctl_c1(const uint8_t *c, uint32_t len) {   // offset of the c1 constants (len: none)
    uint32_t c1 = len;
    if (!ctl_nvn2(c)) memcpy(&c1, c + 0x700, 4);
    else memcpy(&c1, c + 0x40, 4);   // NVN2: after the code and its padding, up to +0x44
    return c1;
}
static const uint8_t *ctl_local(const uint8_t *c) { return c + (ctl_nvn2(c) ? 0x58 : 0x748); }

static int find_stage_scan(const prog_t *p, int nvn_stage, const uint8_t **ctl, const uint8_t **code, uint32_t *len) {
    for (uint32_t i = 0; i < p->nstage && i < 6; i++) {
        const uint8_t *c = p->sd[i].control;
        if (!c || !p->sd[i].data) continue;
        uint32_t st;
        ctl_stage_len(c, &st, len);
        if ((int)st != nvn_stage) continue;
        *ctl = c, *code = (const uint8_t *)(uintptr_t)p->sd[i].data;
        return 1;
    }
    return 0;
}

// Which shader of a program serves each NVN stage, and its code length, per program serial (new at every
// nvnProgramSetShaders): the scan reads two words of every stage's control block, cold guest cache lines, and runs
// several times a draw. Direct-mapped; a collision only scans again.
#define STAGEM_N 512
static struct { uint64_t serial; int8_t idx[6]; uint32_t len[6]; } stagem[STAGEM_N];

static int find_stage(const prog_t *p, int nvn_stage, const uint8_t **ctl, const uint8_t **code, uint32_t *len) {
    if (!p->serial || (unsigned)nvn_stage >= 6) return find_stage_scan(p, nvn_stage, ctl, code, len);
    typeof(stagem[0]) *m = &stagem[p->serial * 0x9E3779B97F4A7C15ull >> 55];
    if (m->serial != p->serial) {
        m->serial = p->serial;
        memset(m->idx, -1, sizeof m->idx);
        for (uint32_t i = 0; i < p->nstage && i < 6; i++) {
            const uint8_t *c = p->sd[i].control;
            if (!c || !p->sd[i].data) continue;
            uint32_t st, len;
            ctl_stage_len(c, &st, &len);
            if (st < 6 && m->idx[st] < 0) m->idx[st] = (int8_t)i, m->len[st] = len;
        }
    }
    int i = m->idx[nvn_stage];
    if (i < 0) return 0;
    *ctl = p->sd[i].control, *code = (const uint8_t *)(uintptr_t)p->sd[i].data, *len = m->len[nvn_stage];
    return 1;
}

// Texture and image types of a stage's first 32 slots, each kept with the handle and the generation and slot token it
// was looked up under: a slot is looked up again, a hash probe, only when the handle changed or its slot may resolve
// otherwise (slot_still).
static struct { uint64_t tex[32], img[32], ttok[32], itok[32]; uint32_t tgen[32], igen[32]; uint8_t tt[32], it[32], ifmt[32]; } AC[8];

// vk_texture_slot_token of the slot a texture or image handle names; an empty handle always resolves to nothing.
static uint64_t token_of(uint64_t h) { return h ? vk_texture_slot_token((int)(h & 0xfffff)) : 1; }

// Whether what handle h resolved to under the global generation *egen and its slot token etok (both read before that
// lookup) still holds at global generation gen: nothing at all changed since, or not its slot (the token, read only when
// the global generation moved, about every 16 draws). A token hit moves *egen to gen.
static int slot_still(uint32_t *egen, uint64_t etok, uint64_t h, uint32_t gen) {
    if (*egen == gen) return 1;
    uint64_t t = token_of(h);
    if (!t || t != etok) return 0;
    *egen = gen;
    return 1;
}
static int shader_memo_on(void);

// The slots of (program, stage) whose types its translation reads (gfx_shader_slot_mask), learned from its first
// shader; ~0u while unknown. The types of other slots select the same shader, so answers_for leaves them zero instead
// of looking up all 64 handles. Direct-mapped by serial and stage; a collision only relearns.
#define SLOTM_N 4096
static struct { uint64_t serial; int st; uint32_t m; } slotm[SLOTM_N];

static uint32_t slotm_idx(uint64_t serial, int st) { return (uint32_t)((serial * 8 + (uint64_t)st) * 0x9E3779B97F4A7C15ull >> 52); }

static uint32_t slot_mask_of(uint64_t serial, int st) {
    const typeof(slotm[0]) *e = &slotm[slotm_idx(serial, st)];
    return serial && e->serial == serial && e->st == st ? e->m : ~0u;
}

static void slot_mask_learn(uint64_t serial, int st, const GfxShader *s) {
    if (!serial || slot_mask_of(serial, st) != ~0u) return;
    slotm[slotm_idx(serial, st)] = (typeof(slotm[0])){serial, st, gfx_shader_slot_mask(s)};
}

static uint8_t type_of_handle(uint64_t h) {   // texture lock held
    view_t v;
    int hv;
    const VkTexImage *img = h ? vk_texture_by_id_locked((int)(h & 0xfffff), &v, &hv) : NULL;
    return img ? (uint8_t)gfx_type_of(img, &v, hv) : 0;
}

// The img_format answer of an image handle: the integer format its view gives it (int_view_format, as get_view binds
// it), else 0 (the translator's default, a float format). Texture lock held.
static uint8_t format_of_handle(uint64_t h) {
    view_t v;
    int hv;
    const VkTexImage *img = h ? vk_texture_by_id_locked((int)(h & 0xfffff), &v, &hv) : NULL;
    VkFormat f = VK_FORMAT_UNDEFINED;
    static int logged;
    if (img && hv && v.set_format && int_view_format(v.format, NULL) && logged < 8)
        logged++, rt_log("vk: image view format: handle %#llx view nvn format %#x on vk format %d -> %d",
                         (unsigned long long)h, v.format, img->format, int_view_format(v.format, NULL));
    if (!img || !hv || !v.set_format || !int_view_format(v.format, &f) || texel_bytes(f) != texel_bytes(img->format) || !texel_bytes(f))
        return 0;
    return (uint8_t)int_view_format(v.format, NULL);
}

// The answers of stage st for the slots in mask; the types of other slots stay zero.
static void answers_for(VkExec *ex, int st, const uint8_t *ctl, uint32_t mask, GfxShaderAnswers *a) {
    memset(a, 0, sizeof *a);
    if (st == NVN_STAGE_VS) vk_pipeline_answers(&ex->pipe, a, NULL);
    else if (st == NVN_STAGE_FS) vk_pipeline_answers(&ex->pipe, NULL, a);
    else
        for (int k = 0; k < 3; k++) {
            uint32_t l;
            memcpy(&l, ctl_local(ctl) + 4 * k, 4);
            a->local[k] = (uint16_t)(l ? l : 1);
        }
    typeof(AC[0]) *c = &AC[st];
    // The generation and tokens are read before the lookups (one that creates an entry changes them) and unlocked: a
    // registration racing with them is ordered after this draw, and a slot stored under older ones is only looked up
    // again.
    uint32_t gen = vk_texture_slots_gen();
    int locked = 0, check = shader_memo_on() == 2;
    for (uint32_t m = mask; m; m &= m - 1) {
        int slot = __builtin_ctz(m);
        uint64_t h = ex->tex[st][slot], hi = ex->img[st][slot];
        int tmiss = c->tex[slot] != h || !c->ttok[slot] || !slot_still(&c->tgen[slot], c->ttok[slot], h, gen);
        int imiss = c->img[slot] != hi || !c->itok[slot] || !slot_still(&c->igen[slot], c->itok[slot], hi, gen);
        uint64_t tt = tmiss ? token_of(h) : 0, it = imiss ? token_of(hi) : 0;
        if ((tmiss || imiss || check) && !locked) vk_texture_lock(), locked = 1, BS.ac_miss++;
        if (tmiss) c->tt[slot] = type_of_handle(h), c->tex[slot] = h, c->ttok[slot] = tt, c->tgen[slot] = gen;
        if (imiss)
            c->it[slot] = type_of_handle(hi), c->ifmt[slot] = format_of_handle(hi), c->img[slot] = hi, c->itok[slot] = it,
            c->igen[slot] = gen;
        if (check && (type_of_handle(h) != c->tt[slot] || type_of_handle(hi) != c->it[slot]))   // NX2NAVI_SHADER_MEMO=check
            rt_fatal("vk_exec: kept texture types of stage %d slot %d differ from a lookup", st, slot);
        a->tex_type[slot] = c->tt[slot], a->img_type[slot] = c->it[slot], a->img_format[slot] = c->ifmt[slot];
    }
    if (locked) vk_texture_unlock();
}

// *present: the program has this stage. Returns NULL when it has not or the translation failed. pending (graphics
// stages, vk_pipeline_async_on): a shader not translated yet is queued and NULL returned with *pending = 1. await: a
// shader queued earlier is waited for (gfx_shader_await) instead.
static int sf_mvprev;   // shader_for asks for the vertex shader's last-frame variant (per-object motion)

static const GfxShader *shader_for(VkExec *ex, int st, int *present, int *pending, int await) {
    const uint8_t *ctl, *code;
    uint32_t len;
    *present = 0;
    if (pending) *pending = 0;
    if (!ex->prog || !find_stage(ex->prog, st, &ctl, &code, &len)) return NULL;
    *present = 1;
    GfxShaderAnswers a;
    uint64_t serial = ex->prog->serial;
    answers_for(ex, st, ctl, slot_mask_of(serial, st), &a);
    a.mvprev = (uint8_t)(sf_mvprev && st == NVN_STAGE_VS);
    ShaderEnt *set = NULL;
    if (serial) {
        uint64_t h = (serial * 0x9E3779B97F4A7C15ull ^ (uint64_t)st) ^ nvn_hm_hash(&a, sizeof a);   // a is zero-padded
        set = shader_tab[(h ^ h >> 29) & (SHADER_SETS - 1)];
        for (int w = 0; w < SHADER_WAYS && set[w].used; w++) {
            ShaderEnt *e = &set[w];
            if (e->serial != serial || e->stage != st || memcmp(&e->a, &a, sizeof a)) continue;
            if (w) {   // to the front
                ShaderEnt hit = *e;
                memmove(&set[1], &set[0], w * sizeof *set);
                set[0] = hit;
            }
            return set[0].s;
        }
    }
    int gs = st == NVN_STAGE_VS ? GFX_STAGE_VS : st == NVN_STAGE_FS ? GFX_STAGE_FS : st == NVN_STAGE_TASK ? GFX_STAGE_TS
           : st == NVN_STAGE_MESH ? GFX_STAGE_MS : GFX_STAGE_CS;
    const GfxShader *s = await ? gfx_shader_await(gs, ctl, code, len, &a)
                       : pending && (vk_pipeline_async_on() || a.mvprev) ? gfx_shader_get_async(gs, ctl, code, len, &a, pending)
                                                           : gfx_shader_get(gs, ctl, code, len, NULL, &a);
    if (pending && *pending) return NULL;
    if (s) slot_mask_learn(serial, st, s);
    if (!s) warn_once(W_SHADER, "a stage %d shader of program %llu failed to translate: draws with it are skipped", st,
                      (unsigned long long)serial);
    else if (set) {   // in front, the least recently used way drops out
        memmove(&set[1], &set[0], (SHADER_WAYS - 1) * sizeof *set);
        set[0] = (ShaderEnt){.serial = serial, .stage = st, .used = 1, .a = a, .s = s};
    }
    return s;
}

// ---- descriptors

static int nvn_stage_of(int gfx_stage) {
    return gfx_stage == GFX_STAGE_VS ? NVN_STAGE_VS : gfx_stage == GFX_STAGE_FS ? NVN_STAGE_FS
         : gfx_stage == GFX_STAGE_TS ? NVN_STAGE_TASK : gfx_stage == GFX_STAGE_MS ? NVN_STAGE_MESH : NVN_STAGE_CS;
}

static VkDescriptorBufferInfo zero_info(uint64_t max) {
    return (VkDescriptorBufferInfo){Z.zbuf, 0, max < ZERO_BYTES ? max : ZERO_BYTES};
}

// vk_mem_count, also kept per (class, use) while a graphics build runs (RS).
static void count_use(int cls, int use, uint64_t n) {
    vk_mem_count(cls, use, n);
    if (!RS.rec) return;
    for (int i = 0; i < RS.ncnt; i++)
        if (RS.cnt[i].cls == cls && RS.cnt[i].use == use) {
            RS.cnt[i].bytes += n;
            return;
        }
    if (RS.ncnt < (int)(sizeof RS.cnt / sizeof RS.cnt[0])) RS.cnt[RS.ncnt++] = (RsCount){cls, use, n};
    else RS.cnt_over = 1;
}

// A buffer descriptor for [addr, addr+size) of guest memory: its backing (a pool's VRAM mirror when it has one) when
// aligned, a copy in transient memory when the backing is host memory with an unaligned offset. 0 when neither works.
// written: the shader stores into it (a mirror's range goes back to guest memory at the end of the submission).
static int bind_range(VkExec *ex, uint64_t addr, uint64_t size, uint64_t align, uint64_t max, int use, int written,
                      VkDescriptorBufferInfo *out) {
    VkMemSpan sp;
    if (!addr || !vk_mem_resolve(addr, 1, &sp)) return 0;
    uint64_t n = size && size < sp.avail ? size : sp.avail;
    if (n > max) n = max;
    if (!n) return 0;
    if (sp.offset % align == 0) {
        VK_MEM_USE(sp);
        *out = (VkDescriptorBufferInfo){sp.buffer, sp.offset, n};
        if (written) vk_mem_gpu_wrote_shader(&sp, n), vk_usnap_drop(addr, n);
        count_use(sp.cls, use, n);
        VK_MEM_BOUND(sp);
        return 1;
    }
    if (sp.alloc) return 0;
    VkBuffer b;
    VkDeviceSize o;
    RS.volatile_ = 1;
    memcpy(scratch(ex, n, &b, &o), (const void *)(uintptr_t)addr, n);
    vk_mem_count(sp.cls, VK_MUSE_CPU_READ, n);
    vk_mem_count(VK_MCLS_SCRATCH, use, n);
    *out = (VkDescriptorBufferInfo){b, o, n};
    return 1;
}

static void c0_table(uint8_t *at, const p_bind_buf b[16]) {
    for (int r = 0; r < 16; r++) {
        uint64_t a = b[r].addr;
        uint32_t n = (uint32_t)b[r].size;
        memcpy(at + 16 * r, &a, 8);
        memcpy(at + 16 * r + 8, &n, 4);
    }
}

// The driver constant bank c0 as GLSLC-compiled shaders read it. Storage-buffer tables {u64 address, u32 size} every
// 16 bytes at 0x110 (vertex), 0x510 (fragment), 0x310 (compute): the translator's GlobalToStorage fallback loads each
// candidate base from here and compares the global pointer against it. Compute: dispatch size at 0x200, the uniform
// buffer table at 0x210 + 16 r. Graphics: the largest point size (Maxwell's limit, GLSLC clamps gl_PointSize to it) at
// 0x610, the base instance at 0x644. NVN2 task and mesh stages: storage tables at 0x670 and 0x770 (the game binds
// slots 0-2 there, and their programs read 0x670.. and 0x770..). Other stages: zeros. Unbound slots read zero, since the
// replay keeps them zero.
// Byte offset of a stage's storage buffer table in c0, -1 for none.
static int c0_ssbo_base(int stage) {
    return stage == NVN_STAGE_VS ? 0x110 : stage == NVN_STAGE_FS ? 0x510 : stage == NVN_STAGE_CS ? 0x310
         : stage == NVN_STAGE_TASK ? 0x670 : stage == NVN_STAGE_MESH ? 0x770 : -1;
}

void vk_c0_fill(uint8_t bank[VK_C0_BYTES], int stage, const p_bind_buf ssbo[16], const p_bind_buf ubo[16],
                const uint32_t dispatch[3], uint32_t base_instance) {
    memset(bank, 0, VK_C0_BYTES);
    int base = c0_ssbo_base(stage);
    if (base < 0) return;
    c0_table(bank + base, ssbo);
    if (stage == NVN_STAGE_TASK || stage == NVN_STAGE_MESH) return;
    if (stage == NVN_STAGE_CS) {
        memcpy(bank + 0x200, dispatch, 12);
        c0_table(bank + 0x210, ubo);
        return;
    }
    const float max_point = 2047.0f;
    memcpy(bank + 0x610, &max_point, 4);
    memcpy(bank + 0x644, &base_instance, 4);
}

static int c0_on(void) {   // NX2NAVI_FIX_C0=0: c0 reads zeros instead of the driver constants
    static int on = -1;
    if (on < 0) {
        on = cfg_bool("NX2NAVI_FIX_C0");
    }
    return on;
}

// c0 from replay scratch; a bank equal to the stage's previous one reuses its range, so it costs no copy.
static void nvn2_cs_tables(const uint8_t *ctl, uint32_t *ubos, uint32_t *ssbos);
static uint32_t nvn2_cs_tables_at(const uint8_t *ctl);
static uint32_t cs_push_bytes;   // the size of the last PushComputeConstants (OP_PUSH_CC), replay thread

static VkDescriptorBufferInfo c0_info(VkExec *ex, int st) {
    int k = st == NVN_STAGE_VS ? 0 : st == NVN_STAGE_FS ? 1 : 2;
    // vk_c0_fill reads the storage slots, and the uniform slots and dispatch size for compute or the base instance for
    // graphics: equal inputs give the same bank, so neither the fill nor the 2 KiB compare is needed.
    typeof(B.c0_in[0]) in;
    memset(&in, 0, sizeof in);
    memcpy(in.ssbo, ex->ssbo[st], sizeof in.ssbo);
    if (st == NVN_STAGE_CS) memcpy(in.ubo, ex->ubo[st], sizeof in.ubo), memcpy(in.dim, c0_dim, sizeof in.dim);
    else in.binst = c0_binst;
    // NVN2 compute: the bank holds the push constants from byte 0x10 (then the program's handle table, which the
    // host does not read: textures and images are bound by slot).
    const uint8_t *pctl, *pcode;
    uint32_t plen;
    int nvn2_push = st == NVN_STAGE_CS && ex->prog && find_stage(ex->prog, st, &pctl, &pcode, &plen) && ctl_nvn2(pctl);
    if (nvn2_push) memcpy(in.push, ex->cs_push, sizeof in.push);
    if (B.c0_valid[k] && !memcmp(&in, &B.c0_in[k], sizeof in))
        return (VkDescriptorBufferInfo){B.c0_buf[k], B.c0_off[k], VK_C0_BYTES};
    B.c0_in[k] = in;
    uint8_t tmp[VK_C0_BYTES];
    vk_c0_fill(tmp, st, ex->ssbo[st], ex->ubo[st], c0_dim, c0_binst);
    if (nvn2_push) {
        // the pushed constants from 0x10, the driver's tables after them (push_ptr, nvn2_cs_tables_at)
        uint32_t nu, ns, ta = nvn2_cs_tables_at(pctl);
        nvn2_cs_tables(pctl, &nu, &ns);
        uint32_t pn = ta - 0x10 < sizeof ex->cs_push ? ta - 0x10 : sizeof ex->cs_push;
        memcpy(tmp + 0x10, ex->cs_push, pn < VK_C0_BYTES - 0x10 ? pn : VK_C0_BYTES - 0x10);
        // Bytes 0-0xb of an NVN2 compute bank are the dispatch's group counts (x, y, z). The fog pass after the
        // integration (passes 0115/0116 of a Kokiri frame) maps each thread to tid / (groups * 16 - 1): with zeros
        // there every thread read the same point. An indirect dispatch leaves them 0 (op_dispatch_indirect).
        memcpy(tmp, c0_dim, 12);
        for (uint32_t k = 0; k < nu + ns && ta + 16 * k + 12 <= VK_C0_BYTES; k++) {
            const p_bind_buf *sb = k < nu ? (k < 18 ? &ex->ubo[st][k] : NULL) : (k - nu < 16 ? &ex->ssbo[st][k - nu] : NULL);
            if (!sb) continue;
            uint32_t sz = (uint32_t)sb->size;
            memcpy(tmp + ta + 16 * k, &sb->addr, 8), memcpy(tmp + ta + 16 * k + 8, &sz, 4);
        }
    }
    if (!B.c0_valid[k] || memcmp(tmp, B.c0[k], VK_C0_BYTES)) {
        memcpy(scratch(ex, VK_C0_BYTES, &B.c0_buf[k], &B.c0_off[k]), tmp, VK_C0_BYTES);
        memcpy(B.c0[k], tmp, VK_C0_BYTES);
        B.c0_valid[k] = 1;
    }
    return (VkDescriptorBufferInfo){B.c0_buf[k], B.c0_off[k], VK_C0_BYTES};
}

// The buffer range limits, read once: the device's properties are a cold cache line on every binding of every draw.
static struct { uint64_t ubo_max, ubo_align, sbo_max, sbo_align; } LIM;

static void limits_init(void) {
    const VkPhysicalDeviceLimits *l = &vk_ctx()->props.limits;
    LIM.ubo_align = l->minUniformBufferOffsetAlignment, LIM.sbo_max = l->maxStorageBufferRange;
    LIM.sbo_align = l->minStorageBufferOffsetAlignment;
    __atomic_store_n(&LIM.ubo_max, l->maxUniformBufferRange, __ATOMIC_RELEASE);
}

// Per-object motion (section at the end): while it records a draw again, these replace the uniform bindings of the
// vertex stage's slots 18-20 (last frame's camera, bones and place) and the fragment stage's slot 3 (its parameters).
static int mo_ovr_on;
static VkDescriptorBufferInfo mo_ovr[2][32];   // [vertex, fragment][slot]; buffer 0: not replaced

static VkDescriptorBufferInfo ubo_info(VkExec *ex, int st, int slot) {
    if (mo_ovr_on && (st == NVN_STAGE_VS || st == NVN_STAGE_FS) && slot >= 0 && slot < 32 && mo_ovr[st][slot].buffer)
        return mo_ovr[st][slot];
    if (!__atomic_load_n(&LIM.ubo_max, __ATOMIC_ACQUIRE)) limits_init();
    uint64_t max = LIM.ubo_max, align = LIM.ubo_align;
    VkDescriptorBufferInfo bi;
    if (slot == 0 && c0_on()) return c0_info(ex, st);
    if (slot == 1 && ex->prog) {   // compiler constants stored after the shader code
        const uint8_t *ctl, *code;
        uint32_t len, c1;
        if (find_stage(ex->prog, st, &ctl, &code, &len)) {
            c1 = ctl_c1(ctl, len);
            if (ctl_nvn2(ctl)) memcpy(&len, ctl + 0x44, 4);   // NVN2: c1 is [+0x40, +0x44) after the code
            if (c1 < len && bind_range(ex, (uint64_t)(uintptr_t)code + c1, len - c1, align, max, VK_MUSE_UBO, 0, &bi))
                return bi;
        }
    } else if (slot >= 3 && slot - 3 < 18) {
        const p_bind_buf *b = &ex->ubo[st][slot - 3];
        if (b->size && vk_usnap_find(b->addr, b->size < max ? b->size : max, align, &bi)) {   // as of the submit call
            count_use(VK_MCLS_SCRATCH, VK_MUSE_UBO, bi.range);
            return bi;
        }
        if (bind_range(ex, b->addr, b->size, align, max, VK_MUSE_UBO, 0, &bi)) return bi;
    }
    return zero_info(max);
}

// The NVN slot of a storage binding. The translator numbers a shader's storage buffers in the order it meets them
// (GfxShaderRes.slot); the NVN slot is the entry of the driver constant buffer c0 that holds the buffer's address
// (cbuf_slot 0, handle = word offset): 16-byte entries at byte 0x110 (vertex), 0x510 (fragment) and 0x310 (compute).
// A compute address at c0 0x210..0x30f is uniform block (offset - 0x210) / 16 read as storage: *ubo is set. -1: the
// address is not in c0 (the shader loads it from a user uniform block: ubo_ptr) or the stage has no known table.
// Contains code Copyright (c) 2026 jhaizhou (MIT; see THIRD_PARTY_NOTICES.md).
static int ssbo_slot(const GfxShaderRes *r, int st, int *ubo) {
    *ubo = 0;
    if (r->cbuf_slot != 0) return -1;
    int off = r->handle * 4, base = c0_ssbo_base(st);
    if (st == NVN_STAGE_CS && off >= 0x210 && off < 0x310 && (off - 0x210) % 16 == 0) {
        *ubo = 1;
        return (off - 0x210) / 16;
    }
    return base >= 0 && off >= base && off < base + 0x100 && (off - base) % 16 == 0 ? (off - base) / 16 : -1;
}

// 1 when the shader loads the storage buffer's address from a user uniform block (cbuf_slot 3.., NVN uniform slot
// cbuf_slot - 3, byte offset handle * 4). Some terrain reads its vertices through one, about 500 draws a frame;
// with a zero buffer they collapse and the ground is missing.
static int ubo_ptr_res(const GfxShaderRes *r) {
    return r->cbuf_slot >= 3 && r->cbuf_slot - 3 < 18;
}

// The inline uniform updates of the submission being replayed, newest last (a ring of UPD_N). op_update_ub writes them
// into the buffer's GPU copy; guest memory gets them only from the write-back when the submission ends. The terrain
// rewrites its pointer block this way before every draw, all at one address, so ubo_ptr reads the latest update that
// covers the pointer, else memory. Payloads stay in the recorded command buffer for the whole replay.
#define UPD_N 4096
static struct {
    uint64_t fence;                      // the submission the entries belong to
    uint32_t n;                          // updates noted in it
    struct { uint64_t lo, hi; const uint8_t *data; } u[UPD_N];
} UPD;

static void upd_note(const VkExec *ex, const p_update_ub *u) {
    if (UPD.fence != ex->fence) UPD.fence = ex->fence, UPD.n = 0;
    UPD.u[UPD.n++ % UPD_N] = (typeof(UPD.u[0])){u->dst, u->dst + u->size, u->data};
}

// 1 when an update of this submission supplied [a, a+n).
static int upd_read(const VkExec *ex, uint64_t a, void *out, size_t n) {
    if (UPD.fence != ex->fence) return 0;
    for (uint32_t i = UPD.n, end = UPD.n > UPD_N ? UPD.n - UPD_N : 0; i-- > end;) {
        const typeof(UPD.u[0]) *e = &UPD.u[i % UPD_N];
        if (a >= e->hi || a + n <= e->lo) continue;
        if (a < e->lo || a + n > e->hi) return 0;   // partly covered: memory
        memcpy(out, e->data + (a - e->lo), n);
        return 1;
    }
    return 0;
}

// That address as the bound block holds it at this point of the replay; 0 when the block is unbound or too short. The
// block has no size field.
static uint64_t ubo_ptr(const VkExec *ex, int st, const GfxShaderRes *r) {
    const p_bind_buf *u = &ex->ubo[st][r->cbuf_slot - 3];
    uint64_t off = (uint64_t)r->handle * 4, a = 0;
    VkMemSpan sp;
    if (!u->addr || u->size < off + 8 || upd_read(ex, u->addr + off, &a, 8)) return a;
    if (vk_mem_resolve(u->addr + off, 8, &sp)) memcpy(&a, (const void *)(uintptr_t)(u->addr + off), 8);
    return a;
}

// NVN2 compute c0 (fitted to the storage offsets of 2549 compute programs against their control sections, round 24):
// from byte 0x10 the constants the game pushed (PushComputeConstants), then a table of uniform buffers and one of
// storage buffers, {u64 address, u32 size} every 16 bytes (counts: bytes 1 and 2 of control +0xb0), then 8-byte entries
// (byte 3 of +0xb0, then byte 2 of +0xb4); +0xb8 (high half) is the end of it all, which gives the pushed size. Entry k
// of the storage part is storage slot k (BindStorageBuffer), entry k of the uniform part uniform slot k. Progs 98 and
// 99 (the compute after the composite: auto exposure) push 0x74 bytes, rounded to 0x80, and read storage entries 9 and 8
// at c0+0x120 / +0x1c0; most programs push nothing and have their tables at 0x10.
static void nvn2_cs_tables(const uint8_t *ctl, uint32_t *ubos, uint32_t *ssbos) {
    uint32_t w;
    memcpy(&w, ctl + 0xb0, 4);
    *ubos = w >> 8 & 0xff, *ssbos = w >> 16 & 0xff;
}

// Where the tables start: after the pushed constants.
static uint32_t nvn2_cs_tables_at(const uint8_t *ctl) {
    uint32_t b0, b4, b8;
    memcpy(&b0, ctl + 0xb0, 4), memcpy(&b4, ctl + 0xb4, 4), memcpy(&b8, ctl + 0xb8, 4);
    uint32_t rest = 16 * ((b0 >> 8 & 0xff) + (b0 >> 16 & 0xff)) + 8 * ((b0 >> 24) + (b4 >> 16 & 0xff)), end = b8 >> 16;
    if (!(b0 >> 8 & 0xffff)) return 0x10 + 1024;   // no tables: all of c0 from 0x10 is the pushed constants
    return end >= 0x10 + rest ? end - rest : 0x10;
}

// An NVN2 compute program that takes a storage buffer's address from c0 (c0_info): an entry of the driver's tables
// (the bound uniform or storage slot), or the {u64 address, u32 size} at that place of the constants the game pushed.
// 1 with *out filled; 0 when the program is not NVN2 or the offset is in neither.
static int push_ptr(const VkExec *ex, int st, const GfxShaderRes *r, p_bind_buf *out) {
    const uint8_t *pctl, *pcode;
    uint32_t plen;
    uint32_t o = (uint32_t)r->handle * 4;   // c0 byte offset
    if (st != NVN_STAGE_CS || r->cbuf_slot != 0 || o < 0x10 || !ex->prog || !find_stage(ex->prog, st, &pctl, &pcode, &plen) ||
        !ctl_nvn2(pctl))
        return 0;
    uint32_t nu, ns, ta = nvn2_cs_tables_at(pctl);
    nvn2_cs_tables(pctl, &nu, &ns);
    if (o < ta) {   // in the pushed constants: a pointer the game passes itself
        uint64_t a;
        uint32_t n = 0;
        if (o - 0x10 + 8 > sizeof ex->cs_push) return 0;
        memcpy(&a, ex->cs_push + (o - 0x10), 8);
        if (o - 0x10 + 12 <= sizeof ex->cs_push) memcpy(&n, ex->cs_push + (o - 0x10) + 8, 4);
        *out = (p_bind_buf){0, 0, a, n < (1u << 30) ? n : 0};   // size 0: up to the end of the backing
        return 1;
    }
    uint32_t t = (o - ta) / 16;
    if ((o - ta) % 16 == 0 && t < nu + ns) {
        *out = t < nu ? (t < 18 ? ex->ubo[st][t] : (p_bind_buf){0}) : (t - nu < 16 ? ex->ssbo[st][t - nu] : (p_bind_buf){0});
        return 1;
    }
    static int logged;   // diagnosis: offsets past the tables
    if (logged++ < 16) {
        uint32_t b0, b4, b8;
        memcpy(&b0, pctl + 0xb0, 4), memcpy(&b4, pctl + 0xb4, 4), memcpy(&b8, pctl + 0xb8, 4);
        rt_log("vk: NVN2 compute c0+%#x is past the tables (prog %llu, +0xb0 %08x +0xb4 %08x +0xb8 %08x, tables at %#x)", o,
               (unsigned long long)ex->prog->serial, b0, b4, b8, ta);
    }
    return 0;
}

// The buffer a storage resource names by where its address lives (GfxShaderRes.ptr aside): an NVN slot of the stage, a
// pointer in a user uniform block (size 0: up to the end of the backing), or an NVN2 compute push pointer. *tmp holds
// the last two. NULL when it names none.
static const p_bind_buf *ssbo_direct(const VkExec *ex, int st, const GfxShaderRes *r, int *ubo, p_bind_buf *tmp) {
    int slot = ssbo_slot(r, st, ubo);
    *tmp = (p_bind_buf){0};
    if (slot >= 0 && slot < 16) return *ubo ? &ex->ubo[st][slot] : &ex->ssbo[st][slot];
    if (ubo_ptr_res(r)) return tmp->addr = ubo_ptr(ex, st, r), tmp;
    return push_ptr(ex, st, r, tmp) ? tmp : NULL;
}

// The address of a storage resource read through a pointer table (GfxShaderRes.ptr): the u64 at offset ptr - 1 of the
// buffer it names, as that buffer holds it at this point of the replay (like ubo_ptr); 0 when the buffer is unbound,
// too short or not backed. Ocarina binds such a table as NVN storage slot 1 of most stages (five pointers at 0-0x20)
// and its vertex, pixel and mesh shaders read their instance and light data through them (PORT-HANDOFF.md 7.17).
static uint64_t ptr_addr(const VkExec *ex, int st, const GfxShaderRes *r) {
    p_bind_buf tmp;
    int ubo;
    const p_bind_buf *b = ssbo_direct(ex, st, r, &ubo, &tmp);
    uint64_t off = r->ptr - 1u, a = 0;
    VkMemSpan sp;
    if (!b || !b->addr || (b->size && b->size < off + 8) || upd_read(ex, b->addr + off, &a, 8)) return a;
    if (vk_mem_resolve(b->addr + off, 8, &sp)) memcpy(&a, (const void *)(uintptr_t)(b->addr + off), 8);
    return a;
}

// vk_dump.c: the guest address a compute storage resource resolves to, 0 when none.
uint64_t vk_exec_cs_storage_addr(const VkExec *ex, const GfxShaderRes *r) {
    p_bind_buf tmp;
    int ubo;
    if (r->ptr) return ptr_addr(ex, NVN_STAGE_CS, r);
    const p_bind_buf *b = ssbo_direct(ex, NVN_STAGE_CS, r, &ubo, &tmp);
    return b ? b->addr : 0;
}

static VkDescriptorBufferInfo ssbo_info(VkExec *ex, int st, const GfxShaderRes *r, int written) {
    // The device address table (vk_gpa.c, translator patch 0030). Mirrored pools are not synced for its readers: a sync
    // of every mirror at a table reader's draw copied stale guest bytes over what the GPU had written into the mirrors
    // earlier in the frame, and Kokiri Forest's lit areas went black. A pool read only through the table is rarely a
    // mirrored one (mirrors are made for pools bound as buffers, which sync then).
    if (r->cbuf_slot == VK_GPA_CB_SLOT) {
        VkDescriptorBufferInfo t = vk_gpa_info();
        return t.buffer ? t : (VkDescriptorBufferInfo){Z.zbuf, 0, VK_WHOLE_SIZE};
    }
    if (!__atomic_load_n(&LIM.ubo_max, __ATOMIC_ACQUIRE)) limits_init();
    uint64_t max = LIM.sbo_max, align = LIM.sbo_align;
    VkDescriptorBufferInfo bi;
    int ubo = 0;
    p_bind_buf up = {0};
    const p_bind_buf *b;
    if (r->ptr) up.addr = ptr_addr(ex, st, r), b = &up;   // size 0: up to the end of the backing
    else b = ssbo_direct(ex, st, r, &ubo, &up);
    if (b) {
        // vk_caps.c: shaders translated for sbuf_align address the buffer from its address rounded down to it.
        uint64_t lead = vk_caps.sbuf_align ? b->addr & (vk_caps.sbuf_align - 1) : 0;
        if (bind_range(ex, b->addr - lead, b->size ? b->size + lead : 0, align, max, ubo ? VK_MUSE_UBO : VK_MUSE_SSBO,
                       written, &bi)) {
            if (written && bi.buffer != VK_NULL_HANDLE) note_written(ex, b->addr - lead, bi.range);
            return bi;
        }
        if (b->addr) warn_once(W_SSBO_ALIGN, "storage buffer %#llx is not backed or not aligned: a zero buffer is bound",
                               (unsigned long long)b->addr);
    } else {
        warn_once(W_SSBO_SLOT, "a storage buffer whose address is at c%d+%#x (stage %d) is not replayed: a zero buffer is bound",
                  r->cbuf_slot, r->handle * 4, st);
        static int logged;   // diagnosis: what c1 of an NVN2 compute program is (its control section's c1 range, the
        const uint8_t *ctl, *code;   // uniform slots the game bound)
        uint32_t len;
        if (logged < 4 && r->cbuf_slot == 1 && ex->prog && find_stage(ex->prog, st, &ctl, &code, &len)) {
            uint32_t c1s, c1e, size;
            memcpy(&size, ctl + 0x38, 4), memcpy(&c1s, ctl + 0x40, 4), memcpy(&c1e, ctl + 0x44, 4);
            char line[512];
            int o = snprintf(line, sizeof line, "vk: c1 storage address, prog %llu: code %#x, c1 [%#x, %#x), len %#x; c1 words:",
                             (unsigned long long)ex->prog->serial, size, c1s, c1e, len);
            for (uint32_t k = 0; k < 4 && c1s + 4 * k + 4 <= len; k++) {
                uint32_t wv;
                memcpy(&wv, code + c1s + 4 * k, 4);
                o += snprintf(line + o, sizeof line - o, " %08x", wv);
            }
            for (int k = 0; k < 6 && o < (int)sizeof line - 40; k++)
                if (ex->ubo[st][k].addr)
                    o += snprintf(line + o, sizeof line - o, " ubo%d=%#llx+%#x", k + 3, (unsigned long long)ex->ubo[st][k].addr,
                                  (unsigned)ex->ubo[st][k].size);
            rt_log("%s", line);
            logged++;
        }
    }
    // A shader that writes it gets the sink: the zero buffer also stands in for missing vertex buffers and read-only
    // storage, and words written into it came back as garbage loop counts (a GPU hang at the first gameplay frame).
    if (written && Z.sink) return (VkDescriptorBufferInfo){Z.sink, 0, max < ZERO_BYTES ? max : ZERO_BYTES};
    return zero_info(max);
}

// ---- feedback loops (maxwell.md 1.9, vk.h VkCtx.feedback)
// NX2navi's G-buffer and terrain shaders sample targets of the pass that renders them, at their own pixel, after earlier
// draws of the pass wrote them (albedo, normals, R32F linear depth). Maxwell reads see those writes; in Vulkan the read
// is undefined without a barrier, and on the RX 9070 it showed as dark screen-aligned blocks on the island ground
// and hid Tulin's sage spirit. Before a draw whose fragment shader samples an attachment of
// the open pass that a draw or clear of the pass wrote since the pass began or the last barrier:
// - VK_FEEDBACK_LAYOUT: the image is learned (VkTexImage.fb_learned; it has the feedback usage since its creation), the
//   pass restarts once with it in ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT, and from then on its passes attach it in that
//   layout, its descriptors carry that layout, the draws that sample it use a pipeline with the feedback flag
//   (VkPipeState.feedback) and a by-region barrier precedes such a draw.
// - VK_FEEDBACK_RESTART: the pass ends and restarts behind the full barrier (vk_exec_pass_begin).
static int in_fbl(const VkExec *ex, VkImage img) {
    for (int i = 0; i < ex->nfbl; i++)
        if (ex->fbl_img[i] == img) return 1;
    return 0;
}

// The image and the levels [l0, l1) and layers [a0, a1) a texture's view covers.
typedef struct { VkImage image; uint32_t l0, l1, a0, a1; } SubRange;

static SubRange sub_of(const VkTexImage *img, const view_t *v, int hv) {
    SubRange r = {img->image, 0, img->levels, 0, img->layers};
    if (hv && v->set_levels) r.l0 = v->base_level, r.l1 = v->base_level + (v->nlevels ? v->nlevels : img->levels);
    if (hv && v->set_layers) r.a0 = v->min_layer, r.a1 = v->min_layer + (v->nlayers ? v->nlayers : img->layers);
    return r;
}

// Attachment slots of the open pass (bit 8 depth) whose subresource r covers.
static uint32_t attached_as(const VkExec *ex, const SubRange *r) {
    uint32_t m = 0;
    if (!r->image) return 0;
    for (uint32_t i = 0; i < 9; i++) {
        const VkExecTarget *t = i < 8 ? &ex->color[i] : &ex->depth;
        if ((i < 8 && i >= ex->pass_ncolor) || t->image != r->image) continue;
        if (t->level < r->l0 || t->level >= r->l1) continue;
        if (!t->is3d && (t->base_layer + t->layers <= r->a0 || t->base_layer >= r->a1)) continue;
        m |= 1u << i;
    }
    return m;
}

// Knobs the draw and dispatch paths test on every call, read once: each cfg lookup walks the settings table
// (cfg.c raw_get), about 0.2 ms a frame of replay in Kokiri Forest when they ran per draw.
static int knob_log_mesh(void) {   // NX2NAVI_LOG_MESH=1: the first mesh draws' state
    static int on = -1;
    if (on < 0) on = rt_knob("LOG_MESH", "0");
    return on;
}
static int knob_tm_dispatch(void) {   // NX2NAVI_TM_DISPATCH: the scene upscaler stand-in's mode
    static int on = -1;
    if (on < 0) on = rt_knob("TM_DISPATCH", "1");
    return on;
}
static int knob_log_cs_binds(void) {   // NX2NAVI_LOG_CS_BINDS set: the compute bindings of each program's first dispatch
    static int on = -1;
    if (on < 0) on = cfg_get("NX2NAVI_LOG_CS_BINDS") != NULL;
    return on;
}

// NX2NAVI_TEX_MEMO (tex_info, feedback_scan): 0 off, 1 on, 2 (=check) computes every hit again and stops on a difference.
static int td_mode(void) {
    static int on = -1;
    if (on < 0) {
        const char *e = cfg_value("NX2NAVI_TEX_MEMO");
        on = !strcmp(e, "check") ? 2 : strcmp(e, "0") != 0;
    }
    return on;
}

// The subresource a texture handle samples, per handle, kept while its slot resolves the same (slot_still): a draw's
// texture bindings change the whole-scan memo below, but the handles they bind were mostly looked up before, in this
// pass or an earlier one. Direct-mapped.
#define FM_N 8192
static struct { uint64_t h, tok; uint32_t gen; SubRange r; } FM[FM_N];

static SubRange sampled_sub(uint64_t h) {   // texture lock held
    view_t v;
    int hv;
    const VkTexImage *img = vk_texture_by_id_locked((int)(h & 0xfffff), &v, &hv);
    return img ? sub_of(img, &v, hv) : (SubRange){0};
}

// Attachments of the open pass the fragment shader fs samples; kept while the shader, the texture bindings and the
// pass stay the same.
static uint32_t feedback_scan(VkExec *ex, const GfxShader *fs) {
    if (ex->fb_fs == fs && ex->fb_serial == ex->tex_serial && ex->fb_pass == ex->pass_serial) return ex->fb_sampled;
    const GfxShaderRes *res;
    int n = gfx_shader_resources(fs, GFX_RES_TEX, &res);
    uint32_t m = 0, gen = vk_texture_slots_gen();   // before any lookup
    int memo = td_mode(), locked = 0;
    for (int r = 0; r < n; r++) {
        // Only bound slots (0..127) are looked at. The texture and sampler pool arrays (res.pool: 32768 and 4096
        // descriptors, bindless handles) have slot -1: walking them ran up to about 100,000 empty iterations per scan,
        // about 3.6 ms of replay per frame in Kokiri Forest, and read tex[FS][0..127] for them by mistake.
        if (res[r].pool || res[r].slot < 0) continue;
        int count = res[r].count ? res[r].count : 1;
        if (count > 128 - res[r].slot) count = 128 - res[r].slot;
        for (int k = 0; k < count; k++) {
            int slot = res[r].slot + k;
            uint64_t h = ex->tex[NVN_STAGE_FS][slot];
            if (!h) continue;
            typeof(FM[0]) *f = &FM[(h * 0x9E3779B97F4A7C15ull) >> 51];
            if (memo && f->h == h && slot_still(&f->gen, f->tok, h, gen)) {
                if (memo == 2) {
                    if (!locked) vk_texture_lock(), locked = 1;
                    SubRange c = sampled_sub(h);
                    if (memcmp(&c, &f->r, sizeof c))
                        rt_fatal("vk_exec: feedback memo of handle %#llx differs from a lookup", (unsigned long long)h);
                }
                m |= attached_as(ex, &f->r);
                continue;
            }
            uint64_t tok = token_of(h);   // before the lookup
            if (!locked) vk_texture_lock(), locked = 1;
            SubRange sr = sampled_sub(h);
            m |= attached_as(ex, &sr);
            if (tok) *f = (typeof(*f)){.h = h, .tok = tok, .gen = gen, .r = sr};
            else f->h = 0;
        }
    }
    if (locked) vk_texture_unlock();
    ex->fb_fs = fs, ex->fb_serial = ex->tex_serial, ex->fb_pass = ex->pass_serial, ex->fb_sampled = m;
    return m;
}

// Before the pipeline lookup of a draw with fragment shader fs: learns and restarts as the mode says. Returns the
// sampled attachments in the feedback layout (the pipeline's feedback flag and feedback_barrier).
static uint32_t feedback_prepare(VkExec *ex, const GfxShader *fs) {
    int mode = vk_ctx()->feedback;
    if (mode == VK_FEEDBACK_OFF || !fs || !ex->in_pass) return 0;
    uint32_t m = feedback_scan(ex, fs);
    uint32_t d = m & ex->fb_dirty;
    if (d) vk_exec_frame.fb_draws++;
    if (d & ~(mode == VK_FEEDBACK_LAYOUT ? ex->fbl_mask : 0)) {
        if (mode == VK_FEEDBACK_LAYOUT)
            for (uint32_t i = 0; i < 9; i++) {
                const VkExecTarget *t = i < 8 ? &ex->color[i] : &ex->depth;
                if ((d >> i & 1) && t->img && t->img->fb_usage) ((VkTexImage *)t->img)->fb_learned = 1;
            }
        vk_exec_pass_end(ex);
        vk_exec_pass_begin(ex);   // learned targets now in the feedback layout; the full barrier cleared fb_dirty
        vk_exec_frame.fb_restarts++;
    }
    return mode == VK_FEEDBACK_LAYOUT ? m & ex->fbl_mask : 0;
}

// By-region barrier inside the pass: earlier attachment writes become visible to this draw's fragment shader reads.
static void feedback_barrier(VkExec *ex) {
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                           .srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT |
                                           VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                           .srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                           .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                           .dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT,
                           .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(ex->cb, &di);
    ex->recorded++;
    ex->fb_dirty = 0;
    vk_exec_frame.fb_barriers++;
}

// Write after read inside the open pass: the GPU orders attachment writes after earlier attachment writes, not after
// earlier draws' texture reads of the same image (feedback layout), so a draw or clear writing a target an earlier
// draw of the pass sampled could land before those reads finish and they read the new value (a band of the picture
// a frame out: the fog's flat colour, a dark shadow band). A by-region barrier from fragment shader reads to
// attachment writes orders them; written: the attachments about to be written (bit i colour i, bit 8 depth).
void vk_exec_feedback_war(VkExec *ex, uint32_t written) {
    if (!(written & ex->fb_read)) return;
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                           .srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                           .srcAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                           .dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT |
                                           VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                           .dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT,
                           .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(ex->cb, &di);
    ex->recorded++;
    ex->fb_read = 0;
    vk_exec_frame.fb_war_barriers++;
}

// The attachments the draw with state s writes: colour targets with a nonzero effective write mask, depth with depth
// writes or the stencil test on.
static uint32_t feedback_writes(const VkExec *ex, const VkPipeState *s) {
    uint32_t w = ex->depth.image && ((s->ds.dt && s->ds.dw) || s->ds.st) ? 0x100u : 0;
    for (uint32_t i = 0; i < ex->pass_ncolor; i++)
        if (ex->color[i].image && vk_pipeline_target_wmask(s, (int)i)) w |= 1u << i;
    return w;
}

// Marks the attachments the draw with state s writes, after ordering them behind earlier draws' reads of them
// (vk_exec_feedback_war); sampled: what this draw itself samples in the feedback layout, which a later write must wait
// for in turn (a draw reading and writing one target is the game's own feedback loop, ordered by region as it is).
static void feedback_written(VkExec *ex, const VkPipeState *s, uint32_t sampled) {
    if (vk_ctx()->feedback == VK_FEEDBACK_OFF) return;
    uint32_t w = feedback_writes(ex, s);
    vk_exec_feedback_war(ex, w);
    ex->fb_dirty |= w;
    ex->fb_read |= sampled;
}

static void tex_info_full(VkExec *ex, const GfxShaderRes *r, uint64_t h, int storage, int separate,
                          VkDescriptorImageInfo *out, VkImage *image);

// Sampled-texture descriptors per (handle, resource type, separate), kept while nothing they derive from changed: the
// slot's resolution and texture entry (slot_still) and the samplers (vk_sampler_gen); the layout is set again on every
// hit from the images in the feedback layout (in_fbl of the image kept with it). The first binding of a slot in a submission (vk_texture_epoch) still makes the slot lookup
// for its storage check; a re-upload writes the same image, so the descriptor stays. Draws bind the same textures
// thousands of times a frame (15,000 lookups a frame in the monsters fight); a hit skips the slot lookup, the view key
// with its hash probe and the sampler lookup. Storage images, which note their binding per submission, are
// not kept. Direct-mapped; texture lock held. NX2NAVI_TEX_MEMO=0 turns it off; =check computes every hit again and stops
// on a difference.
#define TD_N 8192
static struct {
    uint64_t h, tok;
    uint32_t gen, epoch, sgen;
    uint16_t type;
    uint8_t separate, valid;
    VkImage image;
    VkDescriptorImageInfo out;
} TD[TD_N];

static void tex_info_lookup(VkExec *ex, const GfxShaderRes *r, uint64_t h, int storage, int separate,
                            VkDescriptorImageInfo *out) {
    int td_on = td_mode();
    VkImage image;
    if (!h || storage || !td_on) return tex_info_full(ex, r, h, storage, separate, out, &image);
    uint32_t gen = vk_texture_slots_gen(), ep = vk_texture_epoch(), sg = vk_sampler_gen();   // before any lookup
    typeof(TD[0]) *m = &TD[(h * 0x9E3779B97F4A7C15ull ^ (uint64_t)r->type << 7 ^ (uint64_t)separate) >> 51];
    if (m->valid && m->h == h && m->type == r->type && m->separate == separate && m->sgen == sg &&
        slot_still(&m->gen, m->tok, h, gen)) {
        if (m->epoch != ep) {   // the slot's first binding in this submission: only its storage check is due
            vk_texture_by_id_locked((int)(h & 0xfffff), NULL, NULL);
            m->epoch = ep;
        }
        *out = m->out;
        out->imageLayout = m->image && ex->nfbl && in_fbl(ex, m->image) ? VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT
                                                                          : VK_IMAGE_LAYOUT_GENERAL;
        if (td_on == 2) {
            VkDescriptorImageInfo full;
            tex_info_full(ex, r, h, storage, separate, &full, &image);
            if (full.sampler != out->sampler || full.imageView != out->imageView || full.imageLayout != out->imageLayout)
                rt_fatal("vk_exec: texture descriptor memo of handle %#llx type %#x differs from a lookup",
                         (unsigned long long)h, r->type);
        }
        return;
    }
    uint64_t tok = token_of(h);   // before the lookup
    tex_info_full(ex, r, h, storage, separate, out, &image);
    *m = (typeof(*m)){.h = h, .tok = tok, .gen = gen, .epoch = ep, .sgen = sg, .type = r->type,
                      .separate = (uint8_t)separate, .valid = 1, .image = image, .out = *out};
}

static void tex_info_full(VkExec *ex, const GfxShaderRes *r, uint64_t h, int storage, int separate,
                          VkDescriptorImageInfo *out, VkImage *image) {
    int ms = (r->type & 0x200) != 0;
    *image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkSampler samp = VK_NULL_HANDLE;
    VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL;
    if (h) {
        view_t v;
        int hv;
        const VkTexImage *img = vk_texture_by_id_locked((int)(h & 0xfffff), &v, &hv);
        if (img) view = get_view(ex, img, &v, hv, r->type, storage), *image = img->image;
        if (img && storage) vk_texture_storage_bound_locked(img);
        if (img && !storage && ex->nfbl && in_fbl(ex, img->image)) layout = VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT;
        if (!storage && !separate) {
            // A depth texture sampled without a compare reads one texel on the Switch, whatever the sampler's filter:
            // blending neighbouring depths puts reconstructed positions in mid-air at object edges (shadow masks).
            int sid = (int)(h >> 20 & 0xfff);
            int point = img && (img->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) && !(r->type & 0x400) &&
                        vk_exec_view_aspect(img, &v, hv) != VK_IMAGE_ASPECT_STENCIL_BIT;
            samp = point ? vk_sampler_point_by_id_locked(sid)
                 : r->type & 0x400 ? vk_sampler_compare_by_id_locked(sid) : vk_sampler_by_id_locked(sid);
        }
    }
    if (!view) {
        int di = dummy_index(res_view_type(r->type), ms);
        view = storage ? storage_dummy(di, r->format) : Z.img[di].view;
    }
    if (!samp) samp = r->type & 0x400 ? Z.samp_shadow : Z.samp;
    *out = (VkDescriptorImageInfo){separate || storage ? VK_NULL_HANDLE : samp, view, layout};
}

// ---- texture and sampler pools (bindless)
// Shaders that take texture handles from memory (TEX.B and kin) index the whole texture pool: the translator declares
// arrays of VK_POOL_TEX images and VK_POOL_SAMP samplers (index =
// handle & 0xfffff and handle >> 20), which vk_pipeline.c spirv_pool_rebind moves to set 2, one image binding per view
// kind (vk.h). One set serves every such pipeline: entry id of binding k holds texture slot id viewed as kind k (a
// stand-in when the slot is empty or cannot be viewed so), entry id of binding VK_POOL_KINDS sampler slot id. pool_sync
// rewrites the entries whose slot changed (vk_texture_slot_token) and fills the kinds a new pipeline asked for; with
// update-after-bind (descriptorIndexing) a write never invalidates a command buffer being recorded. Each submit's first
// sync also looks every registered slot up once, the freshness check a bound texture gets (vk_texture_by_id_locked).
// The set is one of a ring of POOL_SETS, one per submission: on the Switch the GPU reads the pool as it goes, but it
// runs a submission right after nvnQueueSubmitCommands, so what it reads is the pool as of about the submit. Here the
// GPU runs a submission several milliseconds later, and Ocarina registers slots for its next submission right after
// submitting: with one set, an in-flight submission sampled a texture registered for the next one, the lit materials
// of distant characters read the wrong texture as their light or probe data, wrote NaN and bloom spread it (white
// dialogue frames, a fifth of them; none once each submission waited for its GPU work, NX2NAVI_SUBMIT_FINISH). A
// submission's first pool_sync takes the next set (pool_take_set), waits for the GPU only if that set's last
// submission is still running, and brings it up to date from the change log (every entry written since the set was
// last current; the whole set when the log has wrapped). Writes within a submission go to its own set at once, as
// before.
volatile uint32_t vk_pool_kinds_used;
#define POOL_SETS 4
#define POOL_LOG (1u << 20)   // entries of the change log: binding * VK_POOL_TEX + id, one per write
static struct {
    VkDescriptorSetLayout layout, empty;
    VkDescriptorPool dpool;
    VkDescriptorSet set;               // the current submission's (sets[cur])
    VkDescriptorSet sets[POOL_SETS];
    uint64_t set_fence[POOL_SETS];     // the timeline value of the last submission that bound it
    uint64_t set_synced[POOL_SETS];    // the change-log position up to which it holds every write
    int nsets, cur, all;               // all: the first fill, written to every set
    uint64_t cur_fence;                // the submission sets[cur] serves
    uint32_t *log;                     // POOL_LOG entries
    uint64_t logn;                     // writes so far
    uint64_t *last;                    // per entry: the log position of its newest write (catch-up writes it once, there)
    uint64_t switches, waits, catchup, written, rewrites, catch_ns, sync_ns, n_slots, n_refill, n_fresh;
    uint64_t n_by_reg, n_by_addr;      // refills whose token changed by a registration, by an address generation only
    uint64_t dirty_pos;                // vk_texture_dirty_slots_locked position (0: the next sync walks every slot)
    int ready, uab;
    uint32_t kinds, gen, epoch, sgen;
    int retry;                         // the last sync left slots to the creation budget
    uint64_t *tok;                     // per texture slot: vk_texture_slot_token it was filled under (0: stand-ins)
    VkImageView *view;                 // [kind * VK_POOL_TEX + id]: what the entry holds
    VkSampler *samp;                   // [id]
    uint64_t writes;
} PL;

static VkDescriptorSetLayout pool_layout_make(void) {
    VkCtx *c = vk_ctx();
    VkPhysicalDeviceVulkan12Features f12 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceFeatures2 f2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &f12};
    vkGetPhysicalDeviceFeatures2(c->phys, &f2);   // create_device enables every supported feature
    PL.uab = f12.descriptorBindingSampledImageUpdateAfterBind && f12.descriptorBindingPartiallyBound &&
             f12.descriptorBindingUpdateUnusedWhilePending;
    VkDescriptorSetLayoutBinding b[VK_POOL_KINDS + 1];
    VkDescriptorBindingFlags fl[VK_POOL_KINDS + 1];
    VkDescriptorBindingFlags f = (f12.descriptorBindingPartiallyBound ? VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT : 0) |
        (PL.uab ? VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT : 0);
    for (uint32_t k = 0; k <= VK_POOL_KINDS; k++) {
        b[k] = (VkDescriptorSetLayoutBinding){k, k < VK_POOL_KINDS ? VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE : VK_DESCRIPTOR_TYPE_SAMPLER,
                                              k < VK_POOL_KINDS ? VK_POOL_TEX : VK_POOL_SAMP, VK_SHADER_STAGE_ALL, NULL};
        fl[k] = f;
    }
    VkDescriptorSetLayoutBindingFlagsCreateInfo bf = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
                                                      .bindingCount = VK_POOL_KINDS + 1, .pBindingFlags = fl};
    VkDescriptorSetLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .pNext = &bf,
                                          .flags = PL.uab ? VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT : 0,
                                          .bindingCount = VK_POOL_KINDS + 1, .pBindings = b};
    VkDescriptorSetLayout l;
    if (vkCreateDescriptorSetLayout(c->device, &li, NULL, &l) != VK_SUCCESS) rt_fatal("vk: the texture pool layout failed");
    VkDescriptorPoolSize ps[2] = {{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_POOL_KINDS * VK_POOL_TEX},
                                  {VK_DESCRIPTOR_TYPE_SAMPLER, VK_POOL_SAMP}};
    // POOL_SETS sets, fewer where the device has no room for them (one at the least, as before)
    for (int n = POOL_SETS; n >= 1 && !PL.nsets; n--) {
        VkDescriptorPoolSize psn[2] = {{ps[0].type, ps[0].descriptorCount * (uint32_t)n},
                                       {ps[1].type, ps[1].descriptorCount * (uint32_t)n}};
        VkDescriptorPoolCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                         .flags = PL.uab ? VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT : 0,
                                         .maxSets = (uint32_t)n, .poolSizeCount = 2, .pPoolSizes = psn};
        if (vkCreateDescriptorPool(c->device, &pi, NULL, &PL.dpool) != VK_SUCCESS) continue;
        VkDescriptorSetLayout ls[POOL_SETS];
        for (int i = 0; i < n; i++) ls[i] = l;
        VkDescriptorSetAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = PL.dpool,
                                          .descriptorSetCount = (uint32_t)n, .pSetLayouts = ls};
        if (vkAllocateDescriptorSets(c->device, &ai, PL.sets) == VK_SUCCESS) PL.nsets = n;
        else vkDestroyDescriptorPool(c->device, PL.dpool, NULL), PL.dpool = VK_NULL_HANDLE;
    }
    if (!PL.nsets) rt_fatal("vk: the texture pool set failed");
    PL.set = PL.sets[0];
    PL.log = malloc(sizeof *PL.log * POOL_LOG);
    PL.last = calloc((size_t)VK_POOL_KINDS * VK_POOL_TEX + VK_POOL_SAMP, sizeof *PL.last);
    if (!PL.log || !PL.last) rt_fatal("vk: out of memory");
    PL.tok = calloc(VK_POOL_TEX, sizeof *PL.tok);
    PL.view = calloc((size_t)VK_POOL_KINDS * VK_POOL_TEX, sizeof *PL.view);
    PL.samp = calloc(VK_POOL_SAMP, sizeof *PL.samp);
    if (!PL.tok || !PL.view || !PL.samp) rt_fatal("vk: out of memory");
    rt_log("vk: bindless texture pools: %d kinds x %d textures, %d samplers, %s, %d sets (one per submission in turn)",
           VK_POOL_KINDS, VK_POOL_TEX, VK_POOL_SAMP,
           PL.uab ? "update after bind" : "no update-after-bind on this GPU (entries written while bound are undefined)", PL.nsets);
    return l;
}

static PlatMutex pool_lock_ = PLAT_MUTEX_INIT;

VkDescriptorSetLayout vk_pool_set_layout(void) {
    plat_mutex_lock(&pool_lock_);
    if (!PL.layout) PL.layout = pool_layout_make();
    plat_mutex_unlock(&pool_lock_);
    return PL.layout;
}

VkDescriptorSetLayout vk_empty_set_layout(void) {
    plat_mutex_lock(&pool_lock_);
    if (!PL.empty) {
        VkDescriptorSetLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        if (vkCreateDescriptorSetLayout(vk_ctx()->device, &li, NULL, &PL.empty) != VK_SUCCESS) rt_fatal("vk: empty set layout failed");
    }
    plat_mutex_unlock(&pool_lock_);
    return PL.empty;
}

// The translator's sampler type (GfxShaderRes.type) of pool kind k.
static uint16_t pool_kind_type(int k) {
    static const uint16_t dim[4] = {1, 3, 4, 5};   // 1D, 2D, 3D, CUBE
    return (uint16_t)(dim[k >> 1 & 3] | (k & 1 ? 0x100 : 0) | (k & 8 ? 0x400 : 0));
}

#define POOL_WBATCH 512
#define POOL_CREATE_BUDGET 64   // texture entries a pool sync may create (images and uploads), the rest next time
static VkWriteDescriptorSet pool_w[POOL_WBATCH];
static VkDescriptorImageInfo pool_ii[POOL_WBATCH];
static uint32_t pool_nw;

static void pool_flush(void) {
    if (pool_nw) vkUpdateDescriptorSets(vk_ctx()->device, pool_nw, pool_w, 0, NULL);
    PL.writes += pool_nw;
    pool_nw = 0;
}

static void pool_write_to(VkDescriptorSet set, uint32_t binding, uint32_t id, VkImageView view, VkSampler samp) {
    if (pool_nw == POOL_WBATCH) pool_flush();
    pool_ii[pool_nw] = (VkDescriptorImageInfo){samp, view, VK_IMAGE_LAYOUT_GENERAL};
    pool_w[pool_nw] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set,
                                             .dstBinding = binding, .dstArrayElement = id, .descriptorCount = 1,
                                             .descriptorType = view ? VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE : VK_DESCRIPTOR_TYPE_SAMPLER,
                                             .pImageInfo = &pool_ii[pool_nw]};
    pool_nw++;
}

// An entry whose wanted value PL.view / PL.samp already hold: into the current set, and into the log for the others
// (the first fill goes into every set).
static void pool_write(uint32_t binding, uint32_t id, VkImageView view, VkSampler samp) {
    if (PL.all) {
        for (int i = 0; i < PL.nsets; i++) pool_write_to(PL.sets[i], binding, id, view, samp);
        return;
    }
    pool_write_to(PL.set, binding, id, view, samp);
    uint32_t e = binding * VK_POOL_TEX + id;
    PL.last[e] = PL.logn;
    PL.log[PL.logn++ % POOL_LOG] = e;
}

// Writes entry e (binding * VK_POOL_TEX + id) of set as PL.view / PL.samp hold it now.
static void pool_entry_to(VkDescriptorSet set, uint32_t e) {
    uint32_t binding = e / VK_POOL_TEX, id = e % VK_POOL_TEX;
    if (binding < VK_POOL_KINDS) {
        VkImageView v = PL.view[(size_t)binding * VK_POOL_TEX + id];
        if (v) pool_write_to(set, binding, id, v, VK_NULL_HANDLE);
    } else if (id < VK_POOL_SAMP && PL.samp[id]) {
        pool_write_to(set, binding, id, VK_NULL_HANDLE, PL.samp[id]);
    }
}

// A submission's first pool use: the next set of the ring, once the GPU is done with it. 1 when it switched; *from is
// the log position the set holds writes up to (pool_catch_up brings it level once pool_sync has refreshed the entries:
// a view written into a set must be alive, and the refresh is what replaces the views of destroyed textures).
static int pool_take_set(VkExec *ex, uint64_t *from) {
    if (PL.cur_fence == ex->fence) return 0;
    int i = (PL.cur + 1) % PL.nsets;
    if (PL.nsets > 1) {
        const GfxBackend *b = gfx_backend();
        if (b->fence_done() < PL.set_fence[i]) PL.waits++, b->fence_wait(PL.set_fence[i], UINT64_MAX);
    }
    *from = PL.set_synced[i];
    PL.cur = i, PL.set = PL.sets[i], PL.cur_fence = ex->fence, PL.set_fence[i] = ex->fence;
    return PL.nsets > 1;
}

// The current set gets the entries written into other sets in [from, to) of the log, as PL.view / PL.samp hold them now.
static uint64_t pool_memo_hits;   // pool_fill: slots that shared another slot's lookup (defined with pool_memo below)

static void pool_catch_up(uint64_t from, uint64_t to) {
    VkDescriptorSet set = PL.set;
    if (to - from > POOL_LOG) {   // the log has wrapped since: every entry
        PL.rewrites++;
        for (uint32_t e = 0; e < VK_POOL_KINDS * VK_POOL_TEX + VK_POOL_SAMP; e++) pool_entry_to(set, e);
    } else {
        // each entry once, at its newest write: an entry written again later (also by this submission's own sync,
        // which wrote it into this set already) is skipped at the older positions
        PL.catchup += to - from;
        for (uint64_t q = from; q < to; q++) {
            uint32_t e = PL.log[q % POOL_LOG];
            if (PL.last[e] == q) pool_entry_to(set, e), PL.written++;
        }
    }
    pool_flush();
    PL.switches++;
    if (PL.switches % 3600 == 0) {   // diagnosis: what invalidates the slots (vk_texture.c vk_texture_bucket_stats)
        char bs[320];
        vk_texture_bucket_stats(bs, sizeof bs);
        rt_log("vk: texture pool buckets over 3600 switches: %s", bs);
    }
    if (PL.switches % 3600 == 0)
        rt_log("vk: texture pool sets: %llu switches, %llu waited for the GPU, %.1f log entries and %.1f writes per "
               "catch-up, %llu whole rewrites; %.3f ms catch-up and %.3f ms entry refresh per switch (%.0f slots, %.0f "
               "refilled: %.0f registered, %.0f address changes, %.0f shared a lookup; %.0f checked); %llu identical registrations kept",
               (unsigned long long)PL.switches, (unsigned long long)PL.waits, (double)PL.catchup / (double)PL.switches,
               (double)PL.written / (double)PL.switches, (unsigned long long)PL.rewrites,
               PL.catch_ns / 1e6 / (double)PL.switches, PL.sync_ns / 1e6 / (double)PL.switches,
               (double)PL.n_slots / (double)PL.switches, (double)PL.n_refill / (double)PL.switches,
               (double)PL.n_by_reg / (double)PL.switches, (double)PL.n_by_addr / (double)PL.switches,
               (double)pool_memo_hits / (double)PL.switches,
               (double)PL.n_fresh / (double)PL.switches, (unsigned long long)vk_texture_same_regs());
}

// Entry id of every kind in mask: the slot's view of that kind or the kind's stand-in. Texture lock held. 0: the
// slot's texture was left to a later sync by the creation budget (pool_sync).
// NX2NAVI_SKIP_SHADERS=<hash>,...: draws and dispatches whose shader's SPIR-V has one of these FNV-1a hashes (the "vs"/"fs"/
// "cs" hashes of the frame dump's passes.txt) are skipped (diagnosis: which pass makes an effect).
static int shader_skipped(const GfxShader *sh) {
    static uint64_t list[16];
    static int n = -1;
    static struct { const GfxShader *s; uint8_t skip; } memo[256];
    if (n < 0) {
        n = 0;
        const char *e = cfg_get("NX2NAVI_SKIP_SHADERS");
        for (const char *p = e; p && *p && n < 16;) {
            char *end;
            uint64_t v = strtoull(p, &end, 16);
            if (end == p) break;
            list[n++] = v;
            p = *end ? end + 1 : end;
        }
        if (n) rt_log("vk: NX2NAVI_SKIP_SHADERS: %d shader hashes skipped", n);
    }
    if (!n || !sh) return 0;
    uint32_t i = (uint32_t)(((uintptr_t)sh * 0x9E3779B97F4A7C15ull) >> 56);
    if (memo[i].s == sh) return memo[i].skip;
    uint64_t words = 0, h = 0xcbf29ce484222325ull;   // as vk_dump.c shader_hash
    const uint8_t *b = (const uint8_t *)gfx_shader_spirv(sh, &words);
    for (uint64_t k = 0; b && k < words * 4; k++) h = (h ^ b[k]) * 0x100000001b3ull;
    uint8_t skip = 0;
    for (int k = 0; k < n; k++) skip |= list[k] == h;
    memo[i].s = sh, memo[i].skip = skip;
    return skip;
}

// NX2NAVI_POOL=0: draws and dispatches of pipelines that index the pools are skipped (A/B against bindless shaders).
static int pool_on(void) {
    static int on = -1;
    if (on < 0) on = rt_knob("POOL", "1");
    return on;
}

// Slots that resolve alike share one lookup (vk_texture_slot_khash_locked): Ocarina keeps ~400 descriptors of each
// transient render target registered over its storage, and every finalize and initialize of that target changes their
// tokens, so a refill looked up the same texture ~400 times, ~15,000 lookups a frame in Kokiri Forest. An entry holds the
// views one slot got, for the slots with its hash, address generation and entry generation.
#define POOL_MEMO 4096
static struct { uint64_t kh; uint32_t agen, egen, mask; VkImageView view[VK_POOL_KINDS]; } pool_memo[POOL_MEMO];
static uint64_t pool_memo_hits;

static int pool_fill(VkExec *ex, uint32_t id, uint32_t mask, uint64_t tok) {
    vk_texture_create_deferred_locked();
    uint64_t kh = tok ? vk_texture_slot_khash_locked((int)id) : 0;
    uint32_t ag = (uint32_t)tok;
    typeof(pool_memo[0]) *m = kh ? &pool_memo[(kh >> 20) % POOL_MEMO] : NULL;
    if (m && m->kh == kh && m->agen == ag && m->egen == vk_texture_entry_gen() && (m->mask & mask) == mask) {
        for (int k = 0; k < VK_POOL_KINDS; k++) {
            if (!(mask >> k & 1)) continue;
            VkImageView *e = &PL.view[(size_t)k * VK_POOL_TEX + id];
            if (*e != m->view[k]) *e = m->view[k], pool_write((uint32_t)k, id, m->view[k], VK_NULL_HANDLE);
        }
        pool_memo_hits++;
        return 1;
    }
    for (int k = 0; k < VK_POOL_KINDS; k++) {
        if (!(mask >> k & 1)) continue;
        GfxShaderRes r = {0};
        r.type = pool_kind_type(k);
        VkDescriptorImageInfo ii;
        VkImage img;
        tex_info_full(ex, &r, tok ? id | 1ull << 63 : 0, 0, 1, &ii, &img);
        VkImageView *e = &PL.view[(size_t)k * VK_POOL_TEX + id];
        if (*e != ii.imageView) *e = ii.imageView, pool_write((uint32_t)k, id, ii.imageView, VK_NULL_HANDLE);
    }
    int ok = !vk_texture_create_deferred_locked();
    if (m && ok) {   // the generation after the lookups: an entry they created is part of what they resolved
        uint32_t eg = vk_texture_entry_gen();
        if (m->kh != kh || m->agen != ag || m->egen != eg) m->kh = kh, m->agen = ag, m->egen = eg, m->mask = 0;
        for (int k = 0; k < VK_POOL_KINDS; k++)
            if (mask >> k & 1) m->view[k] = PL.view[(size_t)k * VK_POOL_TEX + id];
        m->mask |= mask;
    }
    return ok;
}

static void pool_sync_entries(VkExec *ex);

static void pool_sync(VkExec *ex) {
    if (!PL.layout) vk_pool_set_layout();
    uint64_t from = 0, pre = PL.logn;
    int switched = pool_take_set(ex, &from);   // every submission its own set, before the shortcut
    uint64_t t0 = kern_now_ns();
    pool_sync_entries(ex);
    uint64_t t1 = kern_now_ns();
    PL.sync_ns += t1 - t0;
    if (switched) {
        vk_texture_lock();
        pool_catch_up(from, pre);   // what the other sets got since; this sync's own writes went into it already
        vk_texture_unlock();
        PL.catch_ns += kern_now_ns() - t1;
    }
    PL.set_synced[PL.cur] = PL.logn;
}

typedef struct { VkExec *ex; uint32_t used; int n, retry; } PoolDirty;

static void pool_dirty_slot(int id, void *arg) {   // vk_texture_dirty_slots_locked: a slot whose token may have changed
    PoolDirty *d = arg;
    if (id >= d->n) return;
    uint64_t t = vk_texture_slot_token(id);
    if (t == PL.tok[id]) return;
    PL.n_refill++;
    if (t >> 32 != PL.tok[id] >> 32) PL.n_by_reg++;
    else PL.n_by_addr++;
    if (pool_fill(d->ex, (uint32_t)id, d->used, t)) PL.tok[id] = t;
    else d->retry = 1;
}

static void pool_sync_entries(VkExec *ex) {
    uint32_t used = __atomic_load_n(&vk_pool_kinds_used, __ATOMIC_RELAXED);
    uint32_t gen = vk_texture_slots_gen(), ep = vk_texture_epoch(), sg = vk_sampler_gen();
    if (PL.ready && !PL.retry && used == PL.kinds && gen == PL.gen && ep == PL.epoch && sg == PL.sgen) return;
    vk_texture_lock();
    uint32_t fresh = used & ~PL.kinds;
    int n = vk_texture_slot_count_locked();
    if (n > VK_POOL_TEX) n = VK_POOL_TEX;
    if (!PL.ready) {   // every entry a stand-in first: partially bound arrays are still read out of range by guests
        PL.all = 1;
        for (uint32_t id = 0; id < VK_POOL_TEX; id++) pool_fill(ex, id, 0xffffu, 0);
        for (uint32_t id = 0; id < VK_POOL_SAMP; id++) PL.samp[id] = Z.samp, pool_write(VK_POOL_KINDS, id, VK_NULL_HANDLE, Z.samp);
        pool_flush();
        PL.all = 0;
        PL.ready = 1, PL.sgen = 0;
    }
    int retry = 0;   // slots whose texture the creation budget left: filled again by the next sync
    if (fresh || gen != PL.gen || ep != PL.epoch || PL.retry) {
        vk_texture_create_budget_locked(POOL_CREATE_BUDGET);
        // Only a slot change since the last sync (no new kinds, no new submission, nothing left over): the slots that
        // changed (vk_texture_dirty_slots_locked), not the whole table
        PoolDirty pd = {ex, used, n, 0};
        int walked = !fresh && ep == PL.epoch && !PL.retry && PL.dirty_pos &&
                     vk_texture_dirty_slots_locked(&PL.dirty_pos, pool_dirty_slot, &pd);
        retry = pd.retry;
        if (!walked) PL.dirty_pos = vk_texture_dirty_pos_locked();
        for (int id = 0; id < n && !walked; id++) {   // slots past n were never registered: stand-ins since the first sync
            uint64_t t = vk_texture_slot_token(id);
            if (t != PL.tok[id]) {
                PL.n_refill++;
                if (t >> 32 != PL.tok[id] >> 32) PL.n_by_reg++;
                else PL.n_by_addr++;
                if (pool_fill(ex, (uint32_t)id, used, t)) PL.tok[id] = t;
                else retry = 1;
            } else if (t && fresh) {
                pool_fill(ex, (uint32_t)id, fresh, t);
            } else if (t && ep != PL.epoch) {
                PL.n_fresh++;
                vk_texture_by_id_locked(id, NULL, NULL);   // the submit's freshness check
            }
        }
        if (!walked) PL.n_slots += (uint64_t)n;
        vk_texture_create_budget_locked(-1);
    }
    PL.retry = retry;
    if (sg != PL.sgen) {
        int ns = vk_sampler_slot_count_locked();
        for (int id = 0; id < VK_POOL_SAMP; id++) {
            VkSampler s = id < ns ? vk_sampler_by_id_locked(id) : VK_NULL_HANDLE;
            if (!s) s = Z.samp;
            if (s != PL.samp[id]) PL.samp[id] = s, pool_write(VK_POOL_KINDS, (uint32_t)id, VK_NULL_HANDLE, s);
        }
    }
    pool_flush();
    vk_texture_unlock();
    PL.kinds = used, PL.gen = gen, PL.epoch = ep, PL.sgen = sg;
}

// ---- depth twins: colour images under a compare sampler in a compute shader
// NX2navi's compute passes sample an R32_SFLOAT texture through a shadow sampler. Maxwell compares any single-channel
// float or unorm texel; Vulkan only formats with SAMPLED_IMAGE_DEPTH_COMPARISON (VUID-vkCmdDispatch-None-06479), which
// AMD's drivers do not give R32_SFLOAT, so the result there is undefined. Such an image gets a depth twin of the same
// size and bits (R32_SFLOAT -> D32_SFLOAT, R16_UNORM -> D16_UNORM), copied through a buffer right before each dispatch
// that samples it (a dispatch is outside any render pass), and the dispatch samples the twin. Draws keep the colour
// image: a copy cannot go inside their pass.
// NX2NAVI_DEPTH_TWIN=0: off.
#define TWINS 8
typedef struct { VkImage src, img; VmaAllocation mem; VkImageView view[2]; uint32_t w, h, layers; VkFormat fmt; } Twin;
static Twin twins[TWINS];
static unsigned twin_next;
static int twin_dispatch;                // op_dispatch*: the resources being bound are a dispatch's
static VkBuffer twin_buf;
static VmaAllocation twin_buf_mem;
static VkDeviceSize twin_buf_cap;

static void twin_free(void *arg) {   // completion thread, after the last submission that may sample it
    Twin *t = arg;
    VkCtx *c = vk_ctx();
    for (int k = 0; k < 2; k++)
        if (t->view[k]) vkDestroyImageView(c->device, t->view[k], NULL);
    vmaDestroyImage(c->vma, t->img, t->mem);
    free(t);
}

typedef struct { VkBuffer buf; VmaAllocation mem; } TwinBuf;
static void twin_buf_free(void *arg) {   // completion thread, after the last submission that may copy through it
    TwinBuf *b = arg;
    vmaDestroyBuffer(vk_ctx()->vma, b->buf, b->mem);
    free(b);
}

static VkFormat twin_format(const VkTexImage *img) {
    static int on = -1;
    if (on < 0) on = rt_knob("DEPTH_TWIN", "1");
    if (!on || (img->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) || img->samples > 1 || img->depth > 1) return VK_FORMAT_UNDEFINED;
    VkFormat df = img->format == VK_FORMAT_R32_SFLOAT ? VK_FORMAT_D32_SFLOAT
                : img->format == VK_FORMAT_R16_UNORM  ? VK_FORMAT_D16_UNORM : VK_FORMAT_UNDEFINED;
    if (!df) return df;
    static int compares[2] = {-1, -1};   // per source format: the driver compares it as it is
    int i = df == VK_FORMAT_D16_UNORM;
    if (compares[i] < 0) {
        VkFormatProperties3 p3 = {.sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3};
        VkFormatProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2, .pNext = &p3};
        vkGetPhysicalDeviceFormatProperties2(vk_ctx()->phys, img->format, &p2);
        compares[i] = (p3.optimalTilingFeatures & VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_DEPTH_COMPARISON_BIT) != 0;
        rt_log("vk_exec: %s under a compare sampler: %s", i ? "R16_UNORM" : "R32_SFLOAT",
               compares[i] ? "compared by the driver" : "sampled through a depth twin in compute passes");
    }
    return compares[i] ? VK_FORMAT_UNDEFINED : df;
}

static void twins_release(VkCtx *c) {
    for (int i = 0; i < TWINS; i++) {
        for (int k = 0; k < 2; k++)
            if (twins[i].view[k]) vkDestroyImageView(c->device, twins[i].view[k], NULL);
        if (twins[i].img) vmaDestroyImage(c->vma, twins[i].img, twins[i].mem);
    }
    memset(twins, 0, sizeof twins);
    if (twin_buf) vmaDestroyBuffer(c->vma, twin_buf, twin_buf_mem);
    twin_buf = VK_NULL_HANDLE, twin_buf_mem = NULL, twin_buf_cap = 0;
}

// The twin of img holding img's current texels, the copies recorded into ex->cb; NULL when img needs none or it cannot
// be made.
static Twin *twin_update(VkExec *ex, const VkTexImage *img) {
    VkFormat df = twin_format(img);
    if (!df) return NULL;
    VkCtx *c = vk_ctx();
    uint32_t bpp = df == VK_FORMAT_D32_SFLOAT ? 4 : 2, layers = img->layers ? img->layers : 1;
    VkDeviceSize bytes = (VkDeviceSize)img->width * img->height * layers * bpp;
    if (bytes > (64u << 20)) return NULL;
    if (bytes > twin_buf_cap) {   // one copy buffer, grown; the old one goes after the submission being recorded
        VkDeviceSize want = twin_buf_cap ? twin_buf_cap * 2 : 1u << 16;
        while (want < bytes) want *= 2;
        VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = want,
                                 .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        VkBuffer nb;
        VmaAllocation nm;
        if (vmaCreateBuffer(c->vma, &bi, &ai, &nb, &nm, NULL) != VK_SUCCESS) return NULL;
        if (twin_buf) {
            TwinBuf *old = malloc(sizeof *old);
            if (!old) rt_fatal("vk: out of memory (depth twin)");
            *old = (TwinBuf){twin_buf, twin_buf_mem};
            vk_completion_add(vk_sync_issued() + 1, twin_buf_free, old);
        }
        twin_buf = nb, twin_buf_mem = nm, twin_buf_cap = want;
    }
    Twin *t = NULL;
    for (int i = 0; i < TWINS && !t; i++)
        if (twins[i].img && twins[i].src == img->image && twins[i].w == img->width && twins[i].h == img->height &&
            twins[i].layers == layers && twins[i].fmt == df)
            t = &twins[i];
    if (!t) {
        t = &twins[twin_next++ % TWINS];
        if (t->img) {   // the submission being recorded may still sample the replaced twin
            Twin *old = malloc(sizeof *old);
            if (!old) rt_fatal("vk: out of memory (depth twin)");
            *old = *t;
            vk_completion_add(vk_sync_issued() + 1, twin_free, old);
        }
        *t = (Twin){.src = img->image, .w = img->width, .h = img->height, .layers = layers, .fmt = df};
        VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = df,
                                .extent = {img->width, img->height, 1}, .mipLevels = 1, .arrayLayers = layers,
                                .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
                                .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        if (vmaCreateImage(c->vma, &ci, &ai, &t->img, &t->mem, NULL) != VK_SUCCESS) {
            *t = (Twin){0};
            return NULL;
        }
        for (int k = 0; k < 2; k++) {
            VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = t->img,
                                        .viewType = k ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D, .format = df,
                                        .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, k ? layers : 1}};
            if (vkCreateImageView(c->device, &vi, NULL, &t->view[k]) != VK_SUCCESS) t->view[k] = VK_NULL_HANDLE;
        }
        static int nlog;
        if (nlog++ < 8)
            rt_log("vk_exec: depth twin %ux%ux%u %s for a compute pass's compare sampler", img->width, img->height,
                   layers, df == VK_FORMAT_D32_SFLOAT ? "D32_SFLOAT" : "D16_UNORM");
    }
    // The source's earlier writes and the twin's earlier readers before the copies; the twin's old texels are dropped.
    VkMemoryBarrier2 m0 = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                           .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT};
    VkImageMemoryBarrier2 ib = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT, .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                .image = t->img, .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, layers}};
    VkDependencyInfo d0 = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &m0,
                           .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &ib};
    vkCmdPipelineBarrier2(ex->cb, &d0);
    VkBufferImageCopy src = {.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, layers},
                             .imageExtent = {img->width, img->height, 1}};
    vkCmdCopyImageToBuffer(ex->cb, img->image, VK_IMAGE_LAYOUT_GENERAL, twin_buf, 1, &src);
    VkMemoryBarrier2 m1 = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                           .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                           .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT};
    VkDependencyInfo d1 = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &m1};
    vkCmdPipelineBarrier2(ex->cb, &d1);
    VkBufferImageCopy dst = {.imageSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, layers},
                             .imageExtent = {img->width, img->height, 1}};
    vkCmdCopyBufferToImage(ex->cb, twin_buf, t->img, VK_IMAGE_LAYOUT_GENERAL, 1, &dst);
    // The twin's texels before the dispatch; the source's next writers and the buffer's next copy after these copies.
    VkMemoryBarrier2 m2 = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                           .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT};
    VkDependencyInfo d2 = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &m2};
    vkCmdPipelineBarrier2(ex->cb, &d2);
    ex->recorded += 5;
    return t;
}

// The image descriptor of slot h; in a dispatch, a compare-sampled colour image the driver cannot compare is swapped
// for its depth twin; in a pass larger than it, the scene's light target, once DLSS scaled it this frame, for DLSS's
// output (vk_dlaa.c).
static void tex_info(VkExec *ex, const GfxShaderRes *r, uint64_t h, int storage, int separate, VkDescriptorImageInfo *out) {
    tex_info_lookup(ex, r, h, storage, separate, out);
    if (h && !storage && vk_dlaa_swap_img && ex->in_pass && ex->pass_w > vk_dlaa_swap_img->width + 1 &&
        vk_texture_by_id_locked((int)(h & 0xfffff), NULL, NULL) == vk_dlaa_swap_img)
        out->imageView = vk_dlaa_swap_view, out->imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    if (!twin_dispatch || storage || separate || !h || !(r->type & 0x400)) return;
    VkImageViewType vt = res_view_type(r->type);
    if (vt != VK_IMAGE_VIEW_TYPE_2D && vt != VK_IMAGE_VIEW_TYPE_2D_ARRAY) return;
    const VkTexImage *img = vk_texture_by_id_locked((int)(h & 0xfffff), NULL, NULL);
    Twin *t = img ? twin_update(ex, img) : NULL;
    VkImageView v = t ? t->view[vt == VK_IMAGE_VIEW_TYPE_2D_ARRAY] : VK_NULL_HANDLE;
    if (v) out->imageView = v, out->imageLayout = VK_IMAGE_LAYOUT_GENERAL;
}

static VkDescriptorSet alloc_set(VkExec *ex, VkDescriptorSetLayout layout) {
    CbRes *r = cb_res(ex);
    VkCtx *c = vk_ctx();
    for (int attempt = 0; attempt < 2; attempt++) {
        if (r->npools) {
            VkDescriptorSetAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                              .descriptorPool = r->pools[r->npools - 1], .descriptorSetCount = 1,
                                              .pSetLayouts = &layout};
            VkDescriptorSet set;
            VkResult res = vkAllocateDescriptorSets(c->device, &ai, &set);
            if (res == VK_SUCCESS) return set;
            if (res != VK_ERROR_OUT_OF_POOL_MEMORY && res != VK_ERROR_FRAGMENTED_POOL) rt_fatal("vk: descriptor set allocation failed (%d)", (int)res);
        }
        static const VkDescriptorPoolSize sizes[] = {
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 4096}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2048},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4096}, {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1024},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1024}, {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 256},
            {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 256}};
        VkDescriptorPoolCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 256,
                                         .poolSizeCount = sizeof sizes / sizeof sizes[0], .pPoolSizes = sizes};
        if (r->npools == r->cpools) r->pools = grow(r->pools, &r->cpools, sizeof *r->pools);
        VkDescriptorPool pool = VK_NULL_HANDLE;
        plat_mutex_lock(&spare_lock);
        if (nspare_pool) pool = spare_pool[--nspare_pool];
        plat_mutex_unlock(&spare_lock);
        if (!pool && vkCreateDescriptorPool(c->device, &pi, NULL, &pool) != VK_SUCCESS) rt_fatal("vk: descriptor pool failed");
        r->pools[r->npools++] = pool;
    }
    rt_fatal("vk: descriptor set allocation failed twice");
    return VK_NULL_HANDLE;
}

static void support_in(const VkExec *ex, float in[3]) {   // what support_fill reads: viewport size, point size
    float k = ex->pass_scale > 1 ? (float)ex->pass_scale : 1.0f;
    in[0] = ex->in_pass && ex->has_vp ? ex->vp[0][2] * k : (float)ex->pass_w;
    in[1] = ex->in_pass && ex->has_vp ? ex->vp[0][3] * k : (float)ex->pass_h;
    in[2] = ex->point_size > 0 ? ex->point_size : 1.0f;
}

// NX2NAVI_PCSS (percent, 0 = off): the pcss field of the support buffer, which translator patch 0010 reads in every
// explicit-LOD compare on a 2D array texture (the sun cascades). x: penumbra width in UV per unit of light-space depth
// between receiver and blocker (0 keeps the game's single compare), y and z: the narrowest and widest disc radius,
// w: the blocker search radius, all in UV of the cascade map. Radii are in UV, so they hold at any NX2NAVI_SHADOW_SCALE.
// NX2NAVI_PCSS_K overrides the depth-to-penumbra factor at 100 percent, for tuning.
static const float *pcss_params(void) {
    static float p[4] = {-1, 0, 0, 0};
    if (p[0] < 0) {
        int pct = cfg_int("NX2NAVI_PCSS");
        pct = pct < 0 ? 0 : pct > 200 ? 200 : pct;
        const char *k = cfg_get("NX2NAVI_PCSS_K");
        float k100 = k && *k ? (float)atof(k) : 0.5f;
        p[0] = k100 * (float)pct / 100.0f;
        p[1] = 0.35f / 960.0f, p[2] = 5.0f / 960.0f, p[3] = 4.0f / 960.0f;
        if (pct) rt_log("vk: soft shadows (PCSS) at %d%% (depth factor %g)", pct, p[0]);
    }
    return p;
}

// The shadow_spread field of the support buffer: the factor translator patch 0011 applies to the lighting shaders' sun
// shadow filter offset (element 56 .xy of their lighting block), 1 / NX2NAVI_SHADOW_SCALE. The game spaces its four
// compares for its own shadow map texels; with maps NX2NAVI_SHADOW_SCALE times larger, the same offset in UV spans that
// many more texels and the near cascade shows four separate copies of each edge. Dividing by the scale keeps the game's
// spacing in texels. The offset is in UV of the shadow map, so the output resolution plays no part. The scale is read
// once, as vk_texture.c shadow_scale reads it: the maps keep the size they were made with.
static const float *shadow_spread_params(void) {
    static float p[4] = {0, 0, 0, 0};
    if (p[0] == 0) {
        int scale = cfg_int("NX2NAVI_SHADOW_SCALE");
        scale = scale < 1 ? 1 : scale > 4 ? 4 : scale;
        p[0] = 1.0f / (float)scale;
    }
    return p;
}

static void support_fill(uint8_t *b, float w, float h, float point_size) {
    memset(b, 0, SUPPORT_BYTES);
    float inv[4] = {w > 0 ? 2.0f / w : 0, h > 0 ? 2.0f / h : 0, 0, 0}, size[4] = {w, h, 0, 0};
    memcpy(b + 144, inv, 16);
    memcpy(b + 160, size, 16);
    const float one = 1.0f;
    memcpy(b + 1392, &point_size, 4);   // point_size (translator patch 0008): vertex shaders that write none output it
    memcpy(b + 1408, pcss_params(), 16);   // pcss (translator patch 0010): soft sun shadows in the cascade compares
    memcpy(b + 1424, shadow_spread_params(), 16);    // shadow_spread (translator patch 0011): sun shadow filter width
    for (int i = 0; i < 73; i++) memcpy(b + 192 + 16 * i, &one, 4);   // render scale 1: the host renders at guest size
    if (vk_hdr_wanted()) {   // read by the HDR tonemap only (shaders/hdr_tonemap.frag), which declares it past the translator's
        float hdr[4];
        vk_hdr_support(hdr);
        memcpy(b + 1440, hdr, 16);
    }
}

// NX2NAVI_BIND_STATS=1: bindings and descriptors per draw of each stage (Task 2.2's choice of the pushed stage), and how
// often a draw repeats the previous one, logged every 20000 draws: pushes recorded and skipped, allocated sets written
// and reused, stage texture-type recomputations (answers_for), draws with the previous draw's program, bound state
// and textures (shaders and pipeline would be the same) and those that also have its buffers and viewport (all
// descriptors would be the same). BS is declared beside DS.
static int bind_stats_on(void) {
    if (BS.on < 0) {
        const char *e = cfg_get("NX2NAVI_BIND_STATS");
        BS.on = e && *e && strcmp(e, "0");
    }
    return BS.on;
}

static void bind_stats(const VkPipeInfo *info) {
    if (!bind_stats_on() || info->bind_point != VK_PIPELINE_BIND_POINT_GRAPHICS) return;
    for (uint32_t i = 0; i < info->n_bindings; i++) {
        const VkPipeBinding *b = &info->bindings[i];
        int fs = info->n_sets > 1 ? b->set : !(b->stages & VK_SHADER_STAGE_VERTEX_BIT);
        BS.bind[fs]++, BS.desc[fs] += b->count ? b->count : 1;
    }
    if (++BS.draws % 20000 == 0) {
        double d = (double)BS.draws;
        rt_log("vk_exec: per draw over %llu draws: vertex %.2f bindings %.2f descriptors, fragment %.2f bindings %.2f descriptors",
               (unsigned long long)BS.draws, BS.bind[0] / d, BS.desc[0] / d, BS.bind[1] / d, BS.desc[1] / d);
        rt_log("vk_exec: per draw: pushes %.3f skipped %.3f, sets written %.3f reused %.3f, texture types %.3f; "
               "same program and state as the draw before %.3f, also same buffers %.3f; descriptor builds skipped %.3f",
               BS.push / d, BS.push_skip / d, BS.set_new / d, BS.set_reuse / d, BS.ac_miss / d, BS.same_prog / d,
               BS.same_all / d, BS.bind_skip / d);
    }
}

// The bound state a draw reads, compared with the previous draw's (NX2NAVI_BIND_STATS only).
static void repeat_stats(const VkExec *ex, const VkPipeInfo *info) {
    static struct {
        const prog_t *prog;
        VkPipeState pipe;
        uint64_t tex[2][128], img[2][32];
        const VkPipeInfo *info;
        p_bind_buf ubo[2][18], ssbo[2][16];
        float vp[6], point_size;
    } last, cur;
    if (!bind_stats_on()) return;
    memset(&cur, 0, sizeof cur);
    cur.prog = ex->prog, cur.pipe = ex->pipe, cur.info = info;
    const int st[2] = {NVN_STAGE_VS, NVN_STAGE_FS};
    for (int k = 0; k < 2; k++) {
        memcpy(cur.tex[k], ex->tex[st[k]], sizeof cur.tex[k]), memcpy(cur.img[k], ex->img[st[k]], sizeof cur.img[k]);
        memcpy(cur.ubo[k], ex->ubo[st[k]], sizeof cur.ubo[k]), memcpy(cur.ssbo[k], ex->ssbo[st[k]], sizeof cur.ssbo[k]);
    }
    memcpy(cur.vp, ex->vp[0], sizeof cur.vp), cur.point_size = ex->point_size;
    size_t prog_part = offsetof(typeof(cur), info);
    if (!memcmp(&cur, &last, prog_part)) {
        BS.same_prog++;
        if (!memcmp((const char *)&cur + prog_part, (const char *)&last + prog_part, sizeof cur - prog_part)) BS.same_all++;
    }
    last = cur;
}

static int push_skip_on(void) {   // NX2NAVI_PUSH_SKIP=0: every graphics draw pushes its set again
    static int on = -1;
    if (on < 0) {
        on = cfg_bool("NX2NAVI_PUSH_SKIP");
    }
    return on;
}

// A graphics draw whose descriptor inputs equal the previous graphics draw's in this command buffer skips building its
// sets: every write would be the same, the push would be skipped and the allocated set reused without a bind (about a
// quarter of field and monsters2 draws, NX2NAVI_BIND_STATS). The inputs, compared and stored in place, are what
// bind_resources and its helpers read beyond the pipeline info's fixed bindings and Z's stand-ins:
// - the info (bindings, layouts), the program and its serial (the compiler-constant range of UBO slot 1);
// - vk_texture_slots_gen, vk_texture_epoch (a slot's image, view and once-per-submit storage check), vk_sampler_gen;
// - vk_mem_gen (buffer and offset of a span, mirror creation and per-submission sync);
// - the images in the feedback layout (the sampled layout);
// - per binding: the support inputs, c0's storage table and base instance, the bound uniform and storage ranges, the
//   texture and image handles (RsList; compared only when a bind op ran since the last draw, bind_gen);
// - storage buffer addresses read from user uniform blocks (RsList.p, ubo_ptr): compared on every draw, since the
//   block's contents change without a bind op.
// The side effects of a build repeat as no-ops in the same command buffer and submission: c0 and support buffers kept
// in B, note_written's ranges, vk_mem_gpu_wrote's write-backs, the per-submission mirror sync and dirty census note,
// the views cached per submission. The census counts are added again. A build that copied guest bytes into transient
// memory (an unaligned host range) is never skipped after: the copy may differ. B.res_valid says the sets bound hold
// the last build; vk_exec_draw_new_cb's memset clears it, and every build clears it until it ends.
// NX2NAVI_BIND_SKIP=0 builds every draw; =check builds every draw and stops when a draw with equal inputs gets other
// writes, census counts or descriptor commands.
static int bind_skip_on(void) {
    if (RS.on < 0) {
        const char *e = cfg_value("NX2NAVI_BIND_SKIP");
        RS.on = !strcmp(e, "check") ? 2 : strcmp(e, "0") != 0;
        if (RS.on != 1) rt_log("vk_exec: descriptor build skip %s", RS.on ? "check" : "off");
    }
    return RS.on;
}

static void rs_add(RsRange *r, uint32_t *n, const VkExec *ex, const void *at, size_t len) {
    uint32_t off = (uint32_t)((const uint8_t *)at - (const uint8_t *)ex);
    if (*n && r[*n - 1].off + r[*n - 1].len == off) r[*n - 1].len += (uint32_t)len;
    else r[(*n)++] = (RsRange){off, (uint32_t)len};
}

static const RsList *rs_list(const VkExec *ex, const VkPipeInfo *info) {
    if (info == rs_last) return rs_last_l;
    if (!rs_lists.klen) nvn_hm_init(&rs_lists, sizeof info);
    RsList *l = nvn_hm_get(&rs_lists, &info);
    if (!l) {
        static RsRange tmp[MAX_DESC];
        RsPtr ptr[RS_PTRS];
        uint32_t n = 0, nd = 0, np = 0;
        int sup = 0, c0 = 0, ok = 1;
        for (uint32_t i = 0; i < info->n_bindings && ok; i++) {
            const VkPipeBinding *b = &info->bindings[i];
            const GfxShaderRes *r = b->res;
            uint32_t cnt = b->count ? b->count : 1;
            if ((nd += cnt) > MAX_DESC) { ok = 0; break; }   // the build stops on it
            int st = nvn_stage_of(b->stage);
            if (b->kind == VK_PIPE_SUPPORT) {
                sup = 1;
            } else if (b->kind == GFX_RES_UBO) {
                for (uint32_t k = 0; k < cnt; k++) {
                    int slot = r->slot + (int)k;
                    if (slot == 0 && c0_on()) c0 = 1, rs_add(tmp, &n, ex, ex->ssbo[st], sizeof ex->ssbo[st]);
                    else if (slot >= 3 && slot - 3 < 18) rs_add(tmp, &n, ex, &ex->ubo[st][slot - 3], sizeof(p_bind_buf));
                }   // slot 1: the program (bind_sig); others: the zero buffer
            } else if (b->kind == GFX_RES_SSBO) {   // every descriptor of the binding is the same range
                int ubo, slot = ssbo_slot(r, st, &ubo), direct = slot >= 0 && slot < 16;
                if (direct) rs_add(tmp, &n, ex, ubo ? &ex->ubo[st][slot] : &ex->ssbo[st][slot], sizeof(p_bind_buf));
                else if (ubo_ptr_res(r)) rs_add(tmp, &n, ex, &ex->ubo[st][r->cbuf_slot - 3], sizeof(p_bind_buf));
                if (r->ptr || (!direct && ubo_ptr_res(r))) {   // an address read from memory: bind_sig reads it again
                    if (np == RS_PTRS) { ok = 0; break; }
                    ptr[np++] = (RsPtr){st, r};
                }
            } else if ((b->kind == GFX_RES_TEX || b->kind == GFX_RES_IMG) && b->type != VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER &&
                       b->type != VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER) {
                int img = b->kind == GFX_RES_IMG, cap = img ? 32 : 128;
                for (uint32_t k = 0; k < cnt; k++) {
                    int slot = r->slot + (int)k;
                    if (slot >= 0 && slot < cap) rs_add(tmp, &n, ex, img ? &ex->img[st][slot] : &ex->tex[st][slot], 8);
                }
            }
        }
        uint64_t bytes = 0;
        for (uint32_t i = 0; i < n; i++) bytes += tmp[i].len;
        if (bytes > RS_BYTES) ok = 0;
        l = malloc(sizeof *l + sizeof *tmp * n);
        if (!l) rt_fatal("vk: out of memory");
        l->n = n, l->bytes = (uint32_t)bytes, l->sup = (uint8_t)sup, l->c0 = (uint8_t)c0, l->ok = (uint8_t)ok;
        l->np = (uint8_t)np;
        memcpy(l->p, ptr, sizeof *ptr * np);
        memcpy(l->r, tmp, sizeof *tmp * n);
        nvn_hm_put(&rs_lists, &info, l);
    }
    rs_last = info, rs_last_l = l;
    return l;
}

// The signature of a graphics draw's descriptor inputs into RS; 1 when it equals the stored one. The bound-slot bytes
// are compared only when a bind op ran since they were stored or the pipeline changed: otherwise they are the same.
// Check mode compares them anyway and stops when the bind op generation said they could not differ but they do.
static int bind_sig(const VkExec *ex, const VkPipeInfo *info) {
    const RsList *l = rs_list(ex, info);
    if (!l->ok) return RS.hn = 0, RS.info = NULL, 0;
    uint64_t *h = RS.h, diff = 0;
    uint32_t n = 0;
#define PUT(v) do { uint64_t v_ = (uint64_t)(v); diff |= h[n] ^ v_; h[n++] = v_; } while (0)
    PUT((uintptr_t)info);
    PUT((uintptr_t)ex->prog);
    PUT(ex->prog->serial);
    PUT((uint64_t)vk_texture_slots_gen() << 32 | vk_texture_epoch());
    PUT((uint64_t)vk_sampler_gen() << 32 | (uint32_t)ex->nfbl);
    PUT(vk_mem_gen() + vk_usnap_gen());   // both only grow
    for (int i = 0; i < ex->nfbl && i < 9; i++) PUT((uintptr_t)ex->fbl_img[i]);
    if (l->sup) {
        float in[3];
        support_in(ex, in);
        uint64_t a;
        uint32_t ps;
        memcpy(&a, in, 8), memcpy(&ps, &in[2], 4);
        PUT(a);
        PUT(ps);
    }
    if (l->c0) PUT(c0_binst);
    for (int i = 0; i < l->np; i++)
        PUT(l->p[i].res->ptr ? ptr_addr(ex, l->p[i].st, l->p[i].res) : ubo_ptr(ex, l->p[i].st, l->p[i].res));
#undef PUT
    diff |= n != RS.hn;
    RS.hn = n;
    int fast = RS.info == info && RS.gen == bind_gen;
    if (fast && RS.on != 2) return !diff;
    const uint8_t *e = (const uint8_t *)ex;
    uint8_t *d = RS.body;
    int bd = 0;
    for (uint32_t i = 0; i < l->n; i++) {
        const RsRange *r = &l->r[i];
        if (!bd && memcmp(d, e + r->off, r->len)) bd = 1;
        if (bd) memcpy(d, e + r->off, r->len);
        d += r->len;
    }
    if (fast && bd) rt_fatal("vk_exec: descriptor build skip check: bound slots changed without a bind op");
    RS.info = info, RS.gen = bind_gen;
    return !diff && !bd;
}

// Writes allocated set s from its packed descriptors d: through the layout's update template, or with the write
// structs w[i0..i1) when it has none.
static void write_set(const VkPipeInfo *info, uint32_t s, VkDescriptorSet set, const VkDescSlot *d, VkWriteDescriptorSet *w,
                      uint32_t i0, uint32_t i1) {
    if (i1 == i0) return;
    if (info->tpl[s]) {
        vk_pipeline_update_tpl_fn()(vk_ctx()->device, set, info->tpl[s], d);
        return;
    }
    for (uint32_t k = i0; k < i1; k++) w[k].dstSet = set;
    vkUpdateDescriptorSets(vk_ctx()->device, i1 - i0, w + i0, 0, NULL);
}

// Writes the pipeline's sets from the bound state: set 0, and set 1 with independent layouts. Each is pushed into the
// command buffer, or allocated, written and bound; a graphics set whose contents equal the last one allocated for that
// set number in this command buffer is reused, and bound again only when another layout or a push came between. A
// graphics push equal to the last one at its set number, with no other layout or bind between, is not recorded.
// The descriptors go into one packed array d, a VkDescSlot per descriptor in binding order, which a set's update
// template reads directly (VkPipeInfo.tpl); write structs are filled only when a set has no template.
static void bind_resources(VkExec *ex, const VkPipeInfo *info) {
    static VkWriteDescriptorSet w[MAX_BINDINGS];
    static VkDescSlot d[MAX_DESC];
    static VkBufferView bv[MAX_DESC];
    if (info->n_bindings > MAX_BINDINGS) rt_fatal("vk: %u descriptor bindings exceed the replay's %d", info->n_bindings, MAX_BINDINGS);
    bind_stats(info);
    int gfx = info->bind_point == VK_PIPELINE_BIND_POINT_GRAPHICS;
    int skip = gfx ? bind_skip_on() : 0, same = 0, was_valid = B.res_valid;
    if (skip) {
        same = bind_sig(ex, info);
        if (same && was_valid && skip == 1) {
            for (int i = 0; i < RS.ncnt; i++) vk_mem_count(RS.cnt[i].cls, RS.cnt[i].use, RS.cnt[i].bytes);
            BS.bind_skip++;
            ex->recorded++;
            return;
        }
        B.res_valid = 0;   // until the build below ends
        if (skip == 2) memcpy(RS.pcnt, RS.cnt, sizeof RS.cnt), RS.pncnt = RS.ncnt;
        RS.rec = 1, RS.ncnt = 0, RS.cnt_over = 0, RS.volatile_ = 0, RS.cmds = 0;
    }
    uint32_t n_sets = info->n_sets ? info->n_sets : 1;
    uint32_t lo[2] = {0, 0}, hi[2] = {0, 0}, d0[2] = {0, 0};   // bindings [lo, hi) and first descriptor per set
    int need_w = 0;                                             // a set without a template: fill the write structs
    for (uint32_t s = 0, i0 = 0; s < n_sets; s++) {
        uint32_t i1 = i0;
        while (i1 < info->n_bindings && (n_sets == 1 || info->bindings[i1].set == s)) i1++;
        lo[s] = i0, hi[s] = i1, i0 = i1;
        if (i1 > lo[s] && !info->tpl[s]) need_w = 1;
    }
    uint32_t nd = 0;
    for (uint32_t i = 0; i < info->n_bindings; i++) {
        const VkPipeBinding *b = &info->bindings[i];
        const GfxShaderRes *r = b->res;
        uint32_t cnt = b->count ? b->count : 1;
        if (nd + cnt > MAX_DESC) rt_fatal("vk: %u descriptors exceed the replay's %d", nd + cnt, MAX_DESC);
        if (n_sets == 2 && i == lo[1]) d0[1] = nd;
        VkDescSlot *o = &d[nd];
        int st = nvn_stage_of(b->stage), kind = 0;   // kind: 0 buffer, 1 image, 2 texel buffer view
        if (b->kind == VK_PIPE_SUPPORT) {
            float in[3];
            support_in(ex, in);
            if (!B.sup_valid || memcmp(in, B.sup_in, sizeof in)) {   // support_fill reads only these
                uint8_t tmp[SUPPORT_BYTES];
                support_fill(tmp, in[0], in[1], in[2]);
                memcpy(B.sup_in, in, sizeof in);
                if (!B.sup_valid || memcmp(tmp, B.support, SUPPORT_BYTES)) {   // a new buffer only when the contents changed
                    VkBuffer sb;
                    VkDeviceSize so;
                    memcpy(scratch(ex, SUPPORT_BYTES, &sb, &so), tmp, SUPPORT_BYTES);
                    memcpy(B.support, tmp, SUPPORT_BYTES);
                    B.sup_buf = sb, B.sup_off = so, B.sup_valid = 1;
                }
            }
            o->buffer = (VkDescriptorBufferInfo){B.sup_buf, B.sup_off, SUPPORT_BYTES};
        } else if (b->kind == GFX_RES_UBO) {
            for (uint32_t k = 0; k < cnt; k++) o[k].buffer = ubo_info(ex, st, r->slot + (int)k);
        } else if (b->kind == GFX_RES_SSBO) {
            for (uint32_t k = 0; k < cnt; k++) o[k].buffer = ssbo_info(ex, st, r, r->flags & 1);
        } else if (b->kind == GFX_RES_TEX || b->kind == GFX_RES_IMG) {
            int img = b->kind == GFX_RES_IMG;
            if (b->type == VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER || b->type == VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER) {
                kind = 2;
                for (uint32_t k = 0; k < cnt; k++) {   // buffer textures are not replayed yet
                    o[k].w[0] = 0, o[k].w[1] = 0, o[k].w[2] = 0;
                    o[k].view = Z.zview;
                    if (need_w) bv[nd + k] = Z.zview;
                }
            } else {
                kind = 1;
                vk_texture_lock();   // tex_info looks up with the texture lock held
                for (uint32_t k = 0; k < cnt; k++) {
                    int slot = r->slot + (int)k;
                    uint64_t h = 0;
                    if (img) h = slot >= 0 && slot < 32 ? ex->img[st][slot] : 0;
                    else h = slot >= 0 && slot < 128 ? ex->tex[st][slot] : 0;
                    tex_info(ex, r, h, img, r->separate, &o[k].image);
                    o[k].w[2] = (uint32_t)o[k].image.imageLayout;   // zero the tail padding
                }
                vk_texture_unlock();
            }
        } else {
            rt_fatal("vk: binding %u has an unknown kind %d", (unsigned)b->binding, (int)b->kind);
        }
        if (need_w) {
            w[i] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                          .dstBinding = b->binding, .descriptorCount = cnt, .descriptorType = b->type};
            if (kind == 0) w[i].pBufferInfo = &o->buffer;   // a VkDescSlot is as large as either info struct
            else if (kind == 1) w[i].pImageInfo = &o->image;
            else w[i].pTexelBufferView = &bv[nd];
        }
        nd += cnt;
    }
    // An empty set 1 (a fragment stage without resources: depth and shadow draws) starts past the last binding, which
    // the loop never reaches: d0[1] stayed 0 and set 0 was pushed with 0 bytes, so the driver read stale descriptors
    if (n_sets == 2 && lo[1] == hi[1]) d0[1] = nd;
    // Under the validation layer, a new graphics layout binds set 1 before set 0 is pushed or bound. The layer (SDK
    // 1.4.363, PipelineLayoutCompatDef::operator==) takes any two INDEPENDENT_SETS layouts as compatible, so
    // vkCmdPushDescriptorSetKHR keeps the push set it made for an earlier layout and drops writes to bindings that
    // layout lacks (false VUID-vkCmdDrawIndexed-None-08114, then an access violation inside the layer). A bind at set 1
    // disturbs set 0 by its canonical id, which drops the layer's push set: the push after it gets a new one.
    int set1_first = 0;
    if (gfx && info->layout != B.ds_layout) {   // another layout: sets bound before may be disturbed
        B.ds_layout = info->layout;
        B.ds_bound[0] = B.ds_bound[1] = VK_NULL_HANDLE;
        B.ds_pushed[0] = B.ds_pushed[1] = 0;
        set1_first = n_sets == 2 && vk_ctx()->validation;
    }
    for (uint32_t k = 0; k < n_sets; k++) {
        uint32_t s = set1_first ? 1 - k : k, i0 = lo[s], i1 = hi[s];
        const VkDescSlot *ds = &d[d0[s]];
        uint32_t dbytes = (uint32_t)(((s + 1 < n_sets ? d0[s + 1] : nd) - d0[s]) * sizeof *d);
        if (s && i1 == i0) {               // no fragment stage: set 1 is empty and unused
            if (set1_first) {              // bound only to drop the layer's push set (comment above)
                VkDescriptorSet set = alloc_set(ex, info->set_layout1);
                vkCmdBindDescriptorSets(ex->cb, info->bind_point, info->layout, 1, 1, &set, 0, NULL);
                B.ds_bound[1] = set, B.ds_pushed[1] = 0;
                RS.cmds++;
            }
            continue;
        }
        if (s ? info->push1 : info->push) {
            uint32_t n = gfx && push_skip_on() ? dbytes : 0;
            if (n && B.ds_pushed[s] && DS.pn[s] == n && !memcmp(DS.psig[s], ds, n)) {
                BS.push_skip++;
                continue;
            }
            if (info->tpl[s]) rec_push_tpl(vk_pipeline_push_tpl_fn(), ex->cb, info->tpl[s], info->layout, s, ds, dbytes);
            else rec_push(vk_pipeline_push_fn(), ex->cb, info->bind_point, info->layout, s, i1 - i0, w + i0);
            BS.push++;
            RS.cmds++;
            if (gfx) {
                B.ds_bound[s] = VK_NULL_HANDLE, B.ds_pushed[s] = n != 0;
                if (n) DS.pn[s] = n, memcpy(DS.psig[s], ds, n);
            }
        } else if (!gfx || s > 1) {
            VkDescriptorSet set = alloc_set(ex, s ? info->set_layout1 : info->set_layout);
            write_set(info, s, set, ds, w, i0, i1);
            vkCmdBindDescriptorSets(ex->cb, info->bind_point, info->layout, s, 1, &set, 0, NULL);
        } else {
            VkDescriptorSetLayout sl = s ? info->set_layout1 : info->set_layout;
            VkDescriptorSet set = B.ds_last[s];
            if (!set || B.ds_last_layout[s] != sl || DS.n[s] != dbytes || memcmp(DS.sig[s], ds, dbytes)) {
                set = alloc_set(ex, sl);
                write_set(info, s, set, ds, w, i0, i1);
                B.ds_last[s] = set, B.ds_last_layout[s] = sl, DS.n[s] = dbytes;
                memcpy(DS.sig[s], ds, dbytes);
                BS.set_new++;
                RS.cmds++;
            } else {
                BS.set_reuse++;
            }
            if (B.ds_bound[s] != set) {
                vkCmdBindDescriptorSets(ex->cb, info->bind_point, info->layout, s, 1, &set, 0, NULL);
                B.ds_bound[s] = set, B.ds_pushed[s] = 0;
                RS.cmds++;
            }
        }
    }
    if (info->pool) {   // set 2: the texture and sampler pools
        pool_sync(ex);
        vkCmdBindDescriptorSets(ex->cb, info->bind_point, info->layout, 2, 1, &PL.set, 0, NULL);
        RS.volatile_ = 1;   // the pools may change under an equal draw: build the next one again
    }
    if (skip) {
        RS.rec = 0;
        // An equal next draw records nothing when every push set holds its push (NX2NAVI_PUSH_SKIP on) and nothing made
        // the build unrepeatable; allocated sets are always bound to the one written last.
        int valid = !RS.volatile_ && !RS.cnt_over && RS.hn;
        for (uint32_t s = 0; s < n_sets; s++)
            if (hi[s] > lo[s] && (s ? info->push1 : info->push) && !B.ds_pushed[s]) valid = 0;
        if (skip == 2) {   // with the pipeline info equal (bind_sig), the packed descriptors determine every write
            uint32_t n = (uint32_t)(nd * sizeof *d);
            int wr = n != RS.ckn || memcmp(RS.ck, d, n);
            int cn = RS.ncnt != RS.pncnt || memcmp(RS.cnt, RS.pcnt, sizeof RS.cnt[0] * (size_t)RS.ncnt);
            if (same && was_valid && (wr || cn || RS.cmds))
                rt_fatal("vk_exec: descriptor build skip check: a draw of program %llu with the inputs of the previous one "
                         "gets %s", (unsigned long long)ex->prog->serial,
                         RS.cmds ? "descriptor commands" : wr ? "other writes" : "other census counts");
            if (same && was_valid) BS.bind_skip++;   // the draws a skip would have spared, compared instead
            RS.ckn = n;
            memcpy(RS.ck, d, n);
        }
        B.res_valid = valid;
    }
    ex->recorded++;
}

// ---- draws

static void bind_pipeline(VkExec *ex, VkPipelineBindPoint bp, VkPipeline p) {
    int k = bp == VK_PIPELINE_BIND_POINT_COMPUTE;
    if (B.pipe[k] == p) return;
    vkCmdBindPipeline(ex->cb, bp, p);
    ex->recorded++;
    B.pipe[k] = p;
    if (!k) B.st_valid = 0;
}

static VkViewport draw_viewport(VkExec *ex) {
    float x = 0, y = 0, w = (float)ex->pass_w, h = (float)ex->pass_h;
    if (ex->has_vp) {
        float k = ex->pass_scale > 1 ? (float)ex->pass_scale : 1.0f;
        x = ex->vp[0][0] * k, y = ex->vp[0][1] * k, w = ex->vp[0][2] * k, h = ex->vp[0][3] * k;
    }
    if (w < 1) w = 1;
    if (h < 1) h = 1;
    VkViewport v = {x, y + h, w, -h, ex->depth_range[0], ex->depth_range[1]};   // the y flip: negative height
    vk_dlaa_viewport(ex, &v);   // the scene's draws move by the DLAA jitter
    return v;
}

static void record_fixed(VkExec *ex) {
    VkCommandBuffer cb = ex->cb;
    if (ex->dirty & VK_EXEC_DIRTY_VIEWPORT) {
        VkViewport v = draw_viewport(ex);
        vkCmdSetViewportWithCount(cb, 1, &v);
        ex->recorded++;
    }
    if (ex->dirty & VK_EXEC_DIRTY_SCISSOR) {
        VkRect2D r = vk_exec_scissor(ex);
        vkCmdSetScissorWithCount(cb, 1, &r);
        ex->recorded++;
    }
    if (ex->dirty & VK_EXEC_DIRTY_DEPTH_BIAS) {
        vkCmdSetDepthBias(cb, ex->poly_offset.units, ex->poly_offset.clamp, ex->poly_offset.factor);
        ex->recorded++;
    }
    if (ex->dirty & VK_EXEC_DIRTY_BLEND_CONST) {
        vkCmdSetBlendConstants(cb, ex->blend_color);
        ex->recorded++;
    }
    if (ex->dirty & VK_EXEC_DIRTY_STENCIL) {
        for (int f = 0; f < 2; f++) {
            VkStencilFaceFlags face = f ? VK_STENCIL_FACE_BACK_BIT : VK_STENCIL_FACE_FRONT_BIT;
            vkCmdSetStencilCompareMask(cb, face, ex->st_vmask[f]);
            vkCmdSetStencilWriteMask(cb, face, ex->st_wmask[f]);
            vkCmdSetStencilReference(cb, face, ex->st_ref[f]);
            ex->recorded += 3;
        }
    }
    ex->dirty = 0;
}

static void bind_vertex_buffers(VkExec *ex) {
    VkBuffer vb[17];
    VkDeviceSize vo[17], vsz[17], vstr[17];
    int vcls[17] = {0};
    void *vpool[17] = {0};
    int dyn = vk_pipeline_vertex_input_dynamic();
    // A stream whose range is the one its bound buffer was resolved from, with no pool or mirror change since
    // (vk_mem_gen, read before any resolve), resolves to that buffer again: most draws change one stream or none.
    uint64_t gen = vk_mem_gen();
    int same_gen = B.vb_valid && B.vgen == gen;
    for (int b = 0; b < 17; b++) {
        vb[b] = Z.zbuf, vo[b] = 0, vsz[b] = ZERO_BYTES, vstr[b] = b < 16 ? ex->pipe.vstream[b].stride : 0;
        if (b == VK_PIPE_ZERO_BINDING) continue;
        const NVNbufferRange *r = &ex->vbuf[b];
        if (same_gen && B.vin[b].address == r->address && B.vin[b].size == r->size) {
            vb[b] = B.vb[b], vo[b] = B.vo[b], vsz[b] = B.vsz[b], vcls[b] = B.vcls[b], vpool[b] = B.vpool[b];
            continue;
        }
        VkMemSpan sp;
        if (!r->address || !vk_mem_resolve(r->address, 1, &sp)) continue;
        VK_MEM_USE(sp);
        uint64_t n = r->size && r->size < sp.avail ? r->size : sp.avail;
        vb[b] = sp.buffer, vo[b] = sp.offset, vsz[b] = n, vcls[b] = sp.cls, vpool[b] = sp.pool;
    }
    memcpy(B.vcls, vcls, sizeof vcls), memcpy(B.vpool, vpool, sizeof vpool);
    memcpy(B.vin, ex->vbuf, sizeof B.vin);
    B.vgen = gen;
    if (B.vb_valid && !memcmp(B.vb, vb, sizeof vb) && !memcmp(B.vo, vo, sizeof vo) && !memcmp(B.vsz, vsz, sizeof vsz) &&
        (dyn || !memcmp(B.vstr, vstr, sizeof vstr)))
        return;
    vkCmdBindVertexBuffers2(ex->cb, 0, 17, vb, vo, vsz, dyn ? NULL : vstr);
    ex->recorded++;
    for (int b = 0; b < 17; b++)
        if (vcls[b]) vk_mem_count(vcls[b], VK_MUSE_VB, vsz[b]);
    if (rt_trace & RT_TRACE_PERF)
        for (int b = 0; b < 17; b++)
            if (vpool[b]) vk_mem_bound_note(vpool[b]);
    memcpy(B.vb, vb, sizeof vb), memcpy(B.vo, vo, sizeof vo), memcpy(B.vsz, vsz, sizeof vsz), memcpy(B.vstr, vstr, sizeof vstr);
    B.vb_valid = 1;
}

// NX2NAVI_PIPE_SKIP: which draws are skipped, rather than waited for, while their shaders or pipeline build.
// strict (0): only draws whose pass targets were all rendered in the previous presented frame; a pass with a target
//   that was not counts as one-shot and waits.
// any (1): also draws in a pass with at least one target rendered in the previous frame (a per-frame depth
//   buffer or g-buffer with a colour target the frame before did not use: half-resolution effect buffers in fights).
//   Passes with none (volumes, LUTs, shadow layers seen for the first time) still wait, and so do passes whose only
//   recent target is the depth buffer while a colour target is rendered for the first time (item icon captures).
// fullscreen (2, default): as any, and full-screen draws into such passes are skipped instead of waiting from their
//   budget. A full-screen pass whose stage libraries are cold in the driver cache took 86 ms on average (up to 194 ms)
//   in a fight on an RX 9070: three frames in a row each spent the 50 ms budget, and some of the draws were skipped
//   after all. Skipping costs the pass for the frames until the pipeline is ready.
static int pipe_skip_mode(void) {
    static int mode = -1;
    if (mode < 0) {
        const char *e = cfg_value("NX2NAVI_PIPE_SKIP");
        mode = !strcmp(e, "strict") || !strcmp(e, "0") ? 0 : !strcmp(e, "fullscreen") || !strcmp(e, "2") ? 2 : 1;
        static const char *const what[3] = {"only draws into per-frame passes (NX2NAVI_PIPE_SKIP=strict)",
                                            "draws into passes with a target the previous frame rendered",
                                            "those and full-screen draws (NX2NAVI_PIPE_SKIP=fullscreen)"};
        rt_log("vk_exec: draws skipped while their pipeline builds: %s", what[mode]);
    }
    return mode;
}

// The previous draw's shaders, kept while its program, bound pipeline state, the vertex and fragment stages' first 32
// texture and image handles and every slot's resolution (vk_texture_slots_gen, read unlocked: a registration racing
// with it is ordered after this draw, as answers_for's own check would order it) stay the same: answers_for reads
// nothing else, so shader_for would return the same shaders. About 65% of field draws repeat them
// (NX2NAVI_BIND_STATS). NX2NAVI_SHADER_MEMO=0 turns it off; =check looks the shaders up on every hit too, and every slot's
// texture type in answers_for, and stops on a difference.
// tex_serial (bumped by every texture and image bind) equal to the key's means the handles were not rebound since
// they were compared, so the 1 KiB compare is skipped; a rebind to the same handles still matches by the compare.
typedef struct {
    const prog_t *prog;
    uint64_t serial;
    uint32_t gen, tex_serial;
    uint64_t tex[2][32], img[2][32];
    VkPipeState pipe;
} MemoKey;
static struct { int on, valid, has_fs; MemoKey k; const GfxShader *vs, *fs; } SM = {-1};

static void memo_key(const VkExec *ex, uint32_t gen, MemoKey *k) {
    k->prog = ex->prog, k->serial = ex->prog->serial, k->gen = gen, k->tex_serial = ex->tex_serial;
    memcpy(k->tex[0], ex->tex[NVN_STAGE_VS], sizeof k->tex[0]), memcpy(k->tex[1], ex->tex[NVN_STAGE_FS], sizeof k->tex[1]);
    memcpy(k->img[0], ex->img[NVN_STAGE_VS], sizeof k->img[0]), memcpy(k->img[1], ex->img[NVN_STAGE_FS], sizeof k->img[1]);
    memcpy(&k->pipe, &ex->pipe, sizeof k->pipe);
}

static int memo_match(const VkExec *ex, uint32_t gen, MemoKey *k) {
    if (k->prog != ex->prog || k->serial != ex->prog->serial || k->gen != gen || memcmp(&k->pipe, &ex->pipe, sizeof k->pipe))
        return 0;
    if (k->tex_serial == ex->tex_serial) return 1;
    if (memcmp(k->tex[0], ex->tex[NVN_STAGE_VS], sizeof k->tex[0]) || memcmp(k->tex[1], ex->tex[NVN_STAGE_FS], sizeof k->tex[1]) ||
        memcmp(k->img[0], ex->img[NVN_STAGE_VS], sizeof k->img[0]) || memcmp(k->img[1], ex->img[NVN_STAGE_FS], sizeof k->img[1]))
        return 0;
    k->tex_serial = ex->tex_serial;
    return 1;
}

static int shader_memo_on(void) {
    if (SM.on < 0) {
        const char *e = cfg_value("NX2NAVI_SHADER_MEMO");
        SM.on = !strcmp(e, "check") ? 2 : strcmp(e, "0") != 0;
    }
    return SM.on;
}

static struct { int on, valid; VkPipeState s; const VkPipeInfo *info; } PM = {-1};

static int pipe_memo_on(void) {
    if (PM.on < 0) PM.on = cfg_bool("NX2NAVI_PIPE_MEMO");
    return PM.on;
}

// Pass, shaders, pipeline, fixed state and resources for a draw. 0: the draw is skipped.
// ---- draw accounting: why draws were not drawn
// Every draw prepare_draw_ex turns away is counted by reason, vertex and mesh draws apart. The first draw of each kind
// skipped for each reason is logged with its program, and every DRAW_STATS_FRAMES presented frames one line sums up
// the draws issued and skipped since the last line (only when something was skipped). A black screen with draws
// skipped for "shader failed" or "pipeline failed" points at the translator or the pipeline; draws issued and still
// nothing on screen point at the shaders' results (NX2NAVI_DUMP).
enum { SK_DISABLED, SK_SCISSOR, SK_NO_PROGRAM, SK_NO_VERTEX_STAGE, SK_SHADER_PENDING, SK_SHADER_FAILED, SK_DLAA,
       SK_PIPE_PENDING, SK_PIPE_FAILED, SK_POOL_OFF, SK_COUNT };
static const char *const sk_names[SK_COUNT] = {
    "rendering disabled (render condition)", "empty scissor", "no program bound",
    "no vertex (or mesh) stage in the program", "shader still translating",
    "shader failed to translate (see gfx_shader: translation failed)", "replaced by the upscaler",
    "pipeline still building", "pipeline could not be created", "uses the texture pools (NX2NAVI_POOL=0)"};
#define DRAW_STATS_FRAMES 600
static struct { uint64_t issued[2], skipped[2][SK_COUNT], frame; uint32_t first[2]; } DRS;

static void draw_stats_tick(void) {
    uint64_t f = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    if (!DRS.frame) DRS.frame = f;
    if (f - DRS.frame < DRAW_STATS_FRAMES) return;
    for (int m = 0; m < 2; m++) {
        uint64_t any = 0;
        for (int k = 0; k < SK_COUNT; k++) any += DRS.skipped[m][k];
        if (!any) continue;
        char line[512];
        int o = snprintf(line, sizeof line, "vk: %s draws over %llu frames: %llu issued, skipped:", m ? "mesh" : "vertex",
                         (unsigned long long)(f - DRS.frame), (unsigned long long)DRS.issued[m]);
        for (int k = 0; k < SK_COUNT && o < (int)sizeof line - 60; k++)
            if (DRS.skipped[m][k]) o += snprintf(line + o, sizeof line - o, " %s %llu;", sk_names[k], (unsigned long long)DRS.skipped[m][k]);
        rt_log("%s", line);
    }
    memset(&DRS, 0, sizeof DRS.issued + sizeof DRS.skipped);
    DRS.frame = f;
}

static int draw_skipped(const VkExec *ex, int mesh, int why) {
    DRS.skipped[mesh][why]++;
    if (!(DRS.first[mesh] >> why & 1)) {
        DRS.first[mesh] |= 1u << why;
        rt_log("vk: first %s draw skipped because: %s (program %llu)", mesh ? "mesh" : "vertex", sk_names[why],
               ex->prog ? (unsigned long long)ex->prog->serial : 0ull);
    }
    draw_stats_tick();
    return 0;
}

// ---- multiplayer: remote players (multiplayer.c)
// A scene draw of local Link's is drawn again for every remote player, with that player's skeleton (c5) and world matrix
// (c7) in place of Link's: the blocks are built in scratch memory (mp_build_remote) and bound through mo_ovr, the
// per-object override mvobj_draw_ already uses. mp_draw_begin, at the end of prepare_draw_ex, asks multiplayer.c whether
// this draw is one of Link's; op_draw and op_draw_idx call mp_repeat after recording the original, which sets the
// overrides, rebuilds the descriptors and re-issues the same draw for each player.
#define MP_C5_MAX (96 * 48)
static struct {
    int ready;                        // the draw is one of Link's and players are waiting for it
    const VkPipeInfo *info;
    int n;                            // the players to draw it again for (their slots: mp_vk_remote_slot, at the repeat)
    int c5_model;                     // local c5's translations are model-space offsets
    uint8_t c5[MP_C5_MAX];            // local Link's skeleton block, as the draw binds it
    uint32_t c5_size;
    uint8_t c7[48];
    int have_c5, have_c7;             // the vertex stage reads them
} MP;
static VkDescriptorBufferInfo mp_ovr_save[2][32];
static uint64_t mp_cam_frame = ~0ull;   // the frame whose camera was taken, so it costs one lookup a frame

// Does any binding of the pipeline's vertex stage read the uniform index? (the translator's cbuf slots are 3 and up)
static int mp_reads_slot(const VkPipeInfo *info, int index) {
    for (uint32_t i = 0; i < info->n_bindings; i++) {
        const VkPipeBinding *b = &info->bindings[i];
        if (b->stage != NVN_STAGE_VS || b->kind != GFX_RES_UBO || !b->res) continue;
        int base = b->res->slot - 3, n = b->count ? b->count : 1;
        if (index >= base && index < base + n) return 1;
    }
    return 0;
}

// The scene camera, once a frame: 1.4.3's deferred pass binds c4 (0x940 bytes), Ocarina's c3 (0x3500 bytes).
// Ocarina's c3 is the 0x3500-byte "scene" block every pass of the frame binds (shadow passes of other lights too), so
// only the frame's first one is taken: it is the main camera's, and taking a later one would sway the nametags.
static void mp_take_camera(VkExec *ex) {
    uint64_t f = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    if (f == mp_cam_frame) return;
    const p_bind_buf *c4 = &ex->ubo[NVN_STAGE_VS][4], *c3 = &ex->ubo[NVN_STAGE_VS][3];
    if (c4->addr && c4->size >= 0x940) {
        const uint8_t *p = vk_usnap_bytes(c4->addr, 0x940);
        if (p) mp_cam_frame = f, mp_observe_camera_ubo(4, p, 0x940);
    } else if (c3->addr && c3->size >= 0x3500) {
        const uint8_t *p = vk_usnap_bytes(c3->addr, 0x3500);
        if (p) mp_cam_frame = f, mp_observe_camera_ubo(3, p, 0x3500);
    }
}

static void mp_draw_begin(VkExec *ex, const VkPipeInfo *info, int mesh, int32_t count) {
    MP.ready = 0, MP.info = info, MP.n = 0, MP.c5_size = 0, MP.have_c5 = MP.have_c7 = 0;
    if (mesh || count <= 0 || !mp_active()) return;
    mp_take_camera(ex);   // before any test that returns: the camera is for the nametags, not only for Link's draws
    const p_bind_buf *c5 = &ex->ubo[NVN_STAGE_VS][5], *c7 = &ex->ubo[NVN_STAGE_VS][7];
    if (!c5->addr || c5->size < 48 || !c7->addr || c7->size < 48) return;
    MP.have_c5 = (uint8_t)mp_reads_slot(info, 5);
    MP.have_c7 = (uint8_t)mp_reads_slot(info, 7);
    if (!MP.have_c5 && !MP.have_c7) return;
    int n = mp_vk_link_draw(c5->addr, (uint32_t)c5->size, c7->addr, (uint32_t)c7->size, &MP.c5_model);
    if (n <= 0) return;
    MP.n = n > 8 ? 8 : n;   // which players: the repeat asks again (mp_vk_remote_slot), the session may change meanwhile
    if (MP.have_c5) {   // the bones as this draw binds them
        MP.c5_size = (uint32_t)c5->size > sizeof MP.c5 ? (uint32_t)sizeof MP.c5 : (uint32_t)c5->size;
        const uint8_t *p = vk_usnap_bytes(c5->addr, MP.c5_size);
        if (p) memcpy(MP.c5, p, MP.c5_size);
        else if (!plat_mem_read(c5->addr, MP.c5, MP.c5_size)) return;
    }
    if (MP.have_c7) {
        const uint8_t *p = vk_usnap_bytes(c7->addr, 48);
        if (p) memcpy(MP.c7, p, 48);
        else if (!plat_mem_read(c7->addr, MP.c7, 48)) return;
    }
    MP.ready = 1;
}

// One player's copy of the draw: their blocks in place of Link's, the descriptors rebuilt around them.
static void mp_build_remote(VkExec *ex, int slot) {
    if (MP.have_c7) {
        uint8_t out[48];
        VkBuffer b;
        VkDeviceSize o;
        mp_vk_build_remote_ubos(slot, NULL, 0, NULL, MP.c7, 48, out, 0);
        memcpy(scratch(ex, 48, &b, &o), out, 48);
        mo_ovr[NVN_STAGE_VS][7] = (VkDescriptorBufferInfo){b, o, 48};
    }
    if (!MP.have_c5 || !MP.c5_size) return;
    VkBuffer b;
    VkDeviceSize o;
    uint8_t *out = scratch(ex, MP.c5_size, &b, &o);
    mp_vk_build_remote_ubos(slot, MP.c5, MP.c5_size, out, NULL, 0, NULL, MP.c5_model);
    mo_ovr[NVN_STAGE_VS][5] = (VkDescriptorBufferInfo){b, o, MP.c5_size};
}

// The draw's copy for each player, right after the original: the same pipeline, pipeline state and vertex buffers
// (which stay bound), the descriptors with the player's blocks.
static void mp_repeat(VkExec *ex, int indexed, uint32_t index_count, uint32_t instance_count, uint32_t first_index,
                      int32_t vertex_offset, uint32_t first_vertex, uint32_t first_instance) {
    if (!MP.ready) return;
    MP.ready = 0;   // this draw's own repeat happens once: a nested one (mvobj_draw never draws) would only repeat it
    for (int i = 0; i < MP.n; i++) {
        int slot = mp_vk_remote_slot(i);
        if (slot < 0) continue;
        memcpy(mp_ovr_save, mo_ovr, sizeof mo_ovr);
        mp_build_remote(ex, slot);
        mo_ovr_on = 1;
        // The skip compares the game's own bind structs (bind_sig), and those are Link's for every copy: without this the
        // second player would be drawn with the first one's blocks.
        B.res_valid = 0;
        bind_resources(ex, MP.info);
        mo_ovr_on = 0;
        memcpy(mo_ovr, mp_ovr_save, sizeof mo_ovr);
        if (indexed) vkCmdDrawIndexed(ex->cb, index_count, instance_count, first_index, vertex_offset, first_instance);
        else vkCmdDraw(ex->cb, index_count, instance_count, first_vertex, first_instance);
        mp_vk_note_re_emitted(1);
        ex->recorded++;
    }
    B.res_valid = 0;   // the sets hold the last player's blocks now: the next draw builds its own
}

// mesh: a mesh draw (op_draw_mesh): the program's mesh and task stages stand in for the vertex stage, and nothing
// vertex-related (vertex buffers, motion vector bookkeeping of vertex shaders) is bound or noted.
static int prepare_draw_ex(VkExec *ex, int prim, int32_t count, int restart, int mesh) {
    if (ex->render_disabled) return vk_cond_draws_skip += ex->has_render_cond, draw_skipped(ex, mesh, SK_DISABLED);
    cb_res(ex);
    dummies_init();
    if (ex->pass_noatt) {   // no attachment: the pass covers this draw's viewport or scissor, reopened when it needs more
        uint32_t w, h;
        vk_exec_noatt_dims(ex, &w, &h);
        if (ex->in_pass && (w > ex->pass_w || h > ex->pass_h)) vk_exec_pass_end(ex);
        if (!ex->in_pass) ex->rt_resolved = 0;
    }
    if (!ex->in_pass) ex->dirty |= VK_EXEC_DIRTY_VIEWPORT | VK_EXEC_DIRTY_SCISSOR;
    vk_exec_pass_begin(ex);
    VkRect2D sc = vk_exec_scissor(ex);
    if (!sc.extent.width || !sc.extent.height) return draw_skipped(ex, mesh, SK_SCISSOR);
    if (!ex->prog) return draw_skipped(ex, mesh, SK_NO_PROGRAM);
    int has_vs, has_fs, vs_wait = 0, fs_wait = 0, memo = mesh ? 0 : shader_memo_on();
    int has_ts = 0, ts_wait = 0, vst = mesh ? NVN_STAGE_MESH : NVN_STAGE_VS;
    const GfxShader *vs, *fs, *ts = NULL;
    uint32_t gen = memo ? vk_texture_slots_gen() : 0;   // before shader_for, whose lookups may bump it
    if (memo && SM.valid && memo_match(ex, gen, &SM.k)) {
        vs = SM.vs, fs = SM.fs, has_vs = 1, has_fs = SM.has_fs;
        if (memo == 2) {
            int hv, hf, wv, wf;
            const GfxShader *v2 = shader_for(ex, NVN_STAGE_VS, &hv, &wv, 0), *f2 = shader_for(ex, NVN_STAGE_FS, &hf, &wf, 0);
            if (v2 != vs || f2 != fs || hf != has_fs || wv || wf)
                rt_fatal("vk_exec: shader memo of program %llu differs from the lookup", (unsigned long long)ex->prog->serial);
        }
    } else {
        SM.valid = 0;
        vs = shader_for(ex, vst, &has_vs, &vs_wait, 0);
        if (mesh) ts = shader_for(ex, NVN_STAGE_TASK, &has_ts, &ts_wait, 0);
        fs = shader_for(ex, NVN_STAGE_FS, &has_fs, &fs_wait, 0);
        if (memo && vs && !vs_wait && !fs_wait && (!has_fs || fs))
            memo_key(ex, gen, &SM.k), SM.vs = vs, SM.fs = fs, SM.has_fs = has_fs, SM.valid = 1;
    }
    if (!has_vs) return draw_skipped(ex, mesh, SK_NO_VERTEX_STAGE);
    // A draw into a target the previous frame did not render (or into none) waits for a pending shader or pipeline:
    // one-shot draws (volumes and LUTs generated once, later only sampled) would leave their target empty for the
    // session. A draw into per-frame targets is skipped until they are ready and comes back next frame.
    int oneshot = -1, skip = pipe_skip_mode();   // -1: not looked up
    if (vs_wait || fs_wait || ts_wait) {
        if (!(oneshot = !vk_exec_targets_recent(ex, skip >= 1)))   // skipped until ready
            return perf_count(PERF_CNT_SHADER_SKIP), draw_skipped(ex, mesh, SK_SHADER_PENDING);
        perf_count(PERF_CNT_ONESHOT_WAIT);
        if (vs_wait) vs = shader_for(ex, vst, &has_vs, NULL, 1);
        if (ts_wait) ts = shader_for(ex, NVN_STAGE_TASK, &has_ts, NULL, 1);
        if (fs_wait) fs = shader_for(ex, NVN_STAGE_FS, &has_fs, NULL, 1);
    }
    if (!vs || (has_fs && !fs) || (has_ts && !ts)) return draw_skipped(ex, mesh, SK_SHADER_FAILED);
    if (shader_skipped(vs) || shader_skipped(fs)) return draw_skipped(ex, mesh, SK_POOL_OFF);   // NX2NAVI_SKIP_SHADERS
    if (vk_dlaa_skip_draw(ex, fs)) return draw_skipped(ex, mesh, SK_DLAA);   // the game's FXAA off, or its scaling replaced by DLSS (vk_dlaa.c)
    uint32_t fbl = feedback_prepare(ex, fs);
    VkPipeState s = ex->pipe;
    s.vs = vs, s.fs = fs, s.ts = ts;
    s.feedback = (uint8_t)((fbl & 0xff ? 1 : 0) | (fbl & 0x100 ? 2 : 0));
    s.prim = (uint8_t)prim;
    s.depth_minus_w = (uint8_t)(nvn_depth_minus_w() != 0);
    s.y_flip = 1;
    s.prim_restart = (uint8_t)(restart != 0);
    const VkPipeInfo *info = NULL;
    int pending;
    // A draw of at most 6 vertices over half the render area or more (a full-screen triangle or quad, a tile) waits for
    // its pipeline: skipping it can leave a black tile. The waits share 50 ms per presented frame
    // (NX2NAVI_PIPE_SKIP=fullscreen skips them instead). A one-shot draw waits without that budget. Everything else is
    // skipped while its pipeline is built.
    static uint64_t wait_frame;
    static uint64_t wait_left_ns;
    uint32_t wait_ms = 0;
    uint64_t t0 = 0;
    if (oneshot > 0) {
        wait_ms = ONESHOT_WAIT_MS;
    } else if (skip < 2 && count <= 6 && (uint64_t)sc.extent.width * sc.extent.height * 2 >= (uint64_t)ex->pass_w * ex->pass_h) {
        uint64_t f = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
        if (f != wait_frame) wait_frame = f, wait_left_ns = 50000000;
        wait_ms = (uint32_t)(wait_left_ns / 1000000);
        if (wait_ms) t0 = kern_now_ns();
    }
    // The previous draw's pipeline while its state repeats (PM, NX2NAVI_PIPE_MEMO=0 turns it off): the cache key is made
    // from the state alone, and a pipeline once found only changes from its fast link to its optimised build, read
    // from the info the cache entry keeps.
    VkPipeline p;
    const char *why = NULL;
    uint64_t w0 = 0;
    if (pipe_memo_on() && PM.valid && !memcmp(&PM.s, &s, sizeof s)) {
        info = PM.info, pending = 0;
        p = __atomic_load_n(&info->pipeline, __ATOMIC_ACQUIRE);
    } else {
        w0 = t0 ? t0 : wait_ms ? kern_now_ns() : 0;   // the clock is read only for draws that may wait
        p = vk_pipeline_graphics_async(&s, &info, &pending, wait_ms);
        if (t0) {
            uint64_t spent = kern_now_ns() - t0;
            wait_left_ns = spent < wait_left_ns ? wait_left_ns - spent : 0;
        }
        why = oneshot > 0 ? "one-shot, shader" : wait_ms ? "full-screen" : NULL;
        if (pending && oneshot < 0 && !vk_exec_targets_recent(ex, skip >= 1)) {
            perf_count(PERF_CNT_ONESHOT_WAIT);
            why = "one-shot";
            if (!w0) w0 = kern_now_ns();
            p = vk_pipeline_graphics_async(&s, &info, &pending, ONESHOT_WAIT_MS);
        }
        PM.valid = !pending && p && info;
        if (PM.valid) PM.s = s, PM.info = info;
    }
    uint64_t waited = w0 ? kern_now_ns() - w0 : 0;
    static int wait_logs;
    if (why && waited >= 2000000 && wait_logs < 200) {
        char tg[512];
        wait_logs++;
        vk_exec_targets_describe(ex, tg, sizeof tg);
        rt_log("vk_exec: frame %llu: a draw waited %.1f ms for its pipeline (%s%s): %u vertices, pass %ux%u, %s",
               (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1, waited / 1e6, why,
               pending ? ", skipped after all" : "", count, ex->pass_w, ex->pass_h, tg);
    }
    if (pending) return draw_skipped(ex, mesh, SK_PIPE_PENDING);   // a background worker is building it: skipped until it is ready
    if (!p || !info) return draw_skipped(ex, mesh, SK_PIPE_FAILED);
    if (info->pool && !pool_on()) return draw_skipped(ex, mesh, SK_POOL_OFF);
    bind_pipeline(ex, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
    if (!B.st_valid || memcmp(&B.st, &s, sizeof s)) {
        vk_pipeline_set_dynamic(ex->cb, &s);
        ex->recorded++;
        B.st = s, B.st_valid = 1;
        if (ex->mark && vk_exec_frame.draws == ex->mark_draws0) ex->mark->first[0] = vs, ex->mark->first[1] = fs;
    }
    if (!mesh) {
        vk_dlaa_draw(ex);
        vk_mvsurvey_draw(ex);
        vk_mvprev_draw(ex);
    }
    record_fixed(ex);
    repeat_stats(ex, info);
    bind_resources(ex, info);
    if (fbl & ex->fb_dirty) feedback_barrier(ex);
    feedback_written(ex, &s, fbl);
    if (!mesh) bind_vertex_buffers(ex);
    if (vk_dump_armed && !mesh) vk_dump_draw(ex, &s, p, info);
    if (vk_hdr_track) vk_hdr_draw(ex, info);
    vk_exec_oq_draw(ex);
    mp_draw_begin(ex, info, mesh, count);   // multiplayer.c: is this draw one of local Link's?
    DRS.issued[mesh]++;
    draw_stats_tick();
    return 1;
}

static int prepare_draw(VkExec *ex, int prim, int32_t count, int restart) {
    return prepare_draw_ex(ex, prim, count, restart, 0);
}

// ---- per-object motion
// The camera pass (vk_dlaa.c mv_pass) gives everything the camera's motion, so a character that moves with the camera,
// or on its own, gets the wrong one and DLSS throws its history away there. Each solid scene draw whose position reads
// an object's own uniform blocks (bones c5, place c7, c6; mvobj_draw_ says which are left out and why) is therefore
// drawn a second time (listed during the replay, MQ, and recorded by mv_pass) with its vertex shader's last-frame
// variant (translator patch 0015: the position again from last frame's blocks, bound at slot bank + 15, and last
// frame's camera, written to a spare output) and the fragment shader of vk_mvobj_fs.h, which writes last frame's window
// position minus this frame's. The draw keeps everything else of the original: vertex and index buffers, the other
// uniform blocks, the jittered viewport and the scene depth, tested with the game's own test and its equal case
// (LEQUAL, GEQUAL) without writes, so only the visible surface writes its vector. Last frame's blocks:
// vk_mvprev_find and mo_prev_block; the camera: vk_dlaa_prev_camera. A draw whose variant or pipeline is still
// building, or whose last-frame blocks are not in the snapshot, keeps the camera's motion. NX2NAVI_DLAA_OBJMV=0 turns it
// off (vk_mvprev.c).
#include "vk_mvobj_fs.h"
static const char MO_FS_RES[] = "{\"key\":\"mvobj\",\"stage\":\"ps\",\"notes\":[],\"cbufs\":[{\"binding\":1,\"slot\":3}],"
                                "\"sbufs\":[],\"textures\":[],\"images\":[]}";
static struct {
    const GfxShader *fs[16];     // per input Location
    int failed;
    uint64_t cand, still, draws, nodata, building, ran, dropped;
    uint64_t ns_check, ns_rec, n_rec, ns_emit, ns_still;
    uint64_t ns_part[3], n_part[3], misses, cyc[3], ncyc, big;   // checks: filter, data and stillness (memo misses), variant and pipeline   // time spent: checks and lookups of every candidate; recording the draws again   // cand: draws whose vertex shader reads c5 or c7
    uint64_t why[5];             // nodata by cause: camera, bones, place, bones copy, place copy
    uint64_t no_out, storage, no_fs;   // variant without the extra output, with storage resources; no fs
    uint64_t other_cam;          // listed draws left out: their camera block is not the scene camera
    uint64_t lk[14][5];          // last-frame lookups per binding index: none, self, side, nearer, refused (another object)
    uint64_t e_draws, e_pipe, e_dyn, e_push, e_pushw, e_sets, e_vb, e_setw, e_setreuse;   // emitted commands (summary)
} MO;

// The motion draws of the frame, kept as a list while the scene replays and recorded into the replay's own command
// buffer by vk_mvobj_execute, through the recorder thread (vk_rec.h) like every other draw: recording them straight
// into a secondary command buffer made the driver's work for each one on the replay thread. What a draw shares with
// the one before it (pipeline, dynamic state, viewport, buffers, descriptors) is recorded once; a push descriptor set
// is pushed again only where a binding changed (pushes update the set in place while its layout stays bound).
typedef struct {
    VkPipeline pipe;
    const VkPipeInfo *info;
    uint32_t st, desc, vb;       // indices into MQ.st, MQ.desc (the draw's packed descriptors) and MQ.vb
    VkDescriptorSet set[2];      // allocated sets; VK_NULL_HANDLE for a pushed one
    VkViewport v;
    VkRect2D sc;
    float bias[3];
    VkBuffer ib;
    VkDeviceSize ib_off;
    VkIndexType it;
    uint32_t count, instances, base_instance;
    int32_t base_vertex;
    uint64_t cam;                // the draw's camera block (c4), this frame's
} MoDraw;
typedef struct { VkBuffer b[17]; VkDeviceSize o[17], sz[17], str[17]; } MoVb;
static struct {
    uint64_t frame, fence;       // the frame and submission the list belongs to
    VkFormat depth_fmt;
    MoDraw *d;
    VkPipeState *st;
    VkDescSlot *desc;
    MoVb *vb;
    int n, cap, nst, cst, nvb, cvb;
    uint32_t ndesc, cdesc;
    NVNbufferRange vin[17];      // the vertex ranges MQ.vb[nvb - 1] was resolved from, under vk_mem_gen vgen
    uint64_t vgen;
    float sup_in[3];             // the support buffer last filled for the list and its inputs
    VkDescriptorBufferInfo sup;
    float par_in[4];             // the fragment shader's parameters last written and their inputs
    VkDescriptorBufferInfo par;
    VkDescriptorSetLayout aset_layout[2];   // the last allocated set per set number, with its descriptors
    VkDescriptorSet aset[2];
    uint32_t aset_n[2];
    VkDescSlot aset_d[2][MAX_DESC];
} MQ;

static void *mo_grow(void *p, int *cap, int need, size_t elem) {
    if (need <= *cap) return p;
    int c = *cap ? *cap : 256;
    while (c < need) c *= 2;
    if (!(p = realloc(p, elem * (size_t)c))) rt_fatal("vk_exec: out of memory");
    *cap = c;
    return p;
}

// The position bank masks of the draws considered (gfx_shader_mvprev_banks), with counts, for the summary.
static struct { uint32_t mask; uint64_t n; int ubo, ssbo, tex; } MM[32];   // ubo, ssbo, tex: the first vertex shader's
static void mo_mask_note(uint32_t banks, const GfxShader *vs) {
    for (int i = 0; i < 32; i++)
        if (MM[i].mask == banks || !MM[i].n) {
            if (!MM[i].n) {
                const GfxShaderRes *r;
                MM[i].ubo = gfx_shader_resources(vs, GFX_RES_UBO, &r), MM[i].ssbo = gfx_shader_resources(vs, GFX_RES_SSBO, &r);
                MM[i].tex = gfx_shader_resources(vs, GFX_RES_TEX, &r);
            }
            MM[i].mask = banks, MM[i].n++;
            return;
        }
}

static void mo_drop(void) {
    if (MQ.n) MO.dropped++;
    MQ.n = MQ.nst = MQ.nvb = 0, MQ.ndesc = 0, MQ.frame = 0;
}

static uint64_t mo_gen = 1;   // the list's snapshot (mo_prev_cached), bumped by mo_begin
static struct { uint32_t mask, chg, n; } MF[16];   // this list's draws per position banks and changed banks
static char fm_last[400];                          // the last executed list's, formatted (vk_mvobj_frame_masks)

// Starts this frame's list (the old one, never executed, is dropped).
static void mo_begin(VkExec *ex, uint64_t frame) {
    mo_drop();
    mo_gen++;
    memset(MF, 0, sizeof MF);
    MQ.frame = frame, MQ.fence = ex->fence, MQ.depth_fmt = ex->depth.format;
    memset(MQ.vin, 0, sizeof MQ.vin);
    MQ.vgen = 0, MQ.sup.buffer = MQ.par.buffer = VK_NULL_HANDLE;
    memset(MQ.aset, 0, sizeof MQ.aset);
}

static const GfxShader *mo_fs(int loc) {
    if (loc < 0 || loc >= 16) return NULL;
    if (!MO.fs[loc]) {
        uint32_t w[sizeof mvobj_fs_spv / 4];
        memcpy(w, mvobj_fs_spv, sizeof w);
        w[MVOBJ_FS_LOCATION_WORD] = (uint32_t)loc;
        MO.fs[loc] = gfx_shader_own(w, sizeof w / 4, MO_FS_RES);
        if (!MO.fs[loc]) MO.failed = 1, rt_log("vk_exec: the per-object motion fragment shader is malformed: off");
    }
    return MO.fs[loc];
}

// The write of binding b from its packed descriptors.
static VkWriteDescriptorSet mo_write(const VkPipeBinding *b, const VkDescSlot *d) {
    VkWriteDescriptorSet w = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = b->binding,
                              .descriptorCount = b->count ? b->count : 1, .descriptorType = b->type};
    if (b->type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER || b->type == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE ||
        b->type == VK_DESCRIPTOR_TYPE_SAMPLER)
        w.pImageInfo = &d->image;
    else
        w.pBufferInfo = &d->buffer;   // VkDescSlot is 24 bytes, as VkDescriptorBufferInfo: arrays line up
    return w;
}

// The descriptors of a motion draw, packed into MQ.desc at q->desc without the binding caches of the command buffer
// being replayed (B, DS, RS), which then stay true for its next draw: uniform buffers (through ubo_info, so with
// mo_ovr), the support buffer and sampled textures. An allocated set is written here (or the last one reused when it
// would hold the same); a pushed set is pushed by mo_emit. 0 for a pipeline with storage buffers, images or texel
// buffers (not drawn again).
static int mo_desc(VkExec *ex, const VkPipeInfo *info, MoDraw *q) {
    if (info->n_bindings > MAX_BINDINGS || info->pool) return 0;
    uint32_t total = 0;
    for (uint32_t i = 0; i < info->n_bindings; i++) total += info->bindings[i].count ? info->bindings[i].count : 1;
    if (total > MAX_DESC) return 0;
    int cdesc = (int)MQ.cdesc;
    MQ.desc = mo_grow(MQ.desc, &cdesc, (int)(MQ.ndesc + total), sizeof *MQ.desc);
    MQ.cdesc = (uint32_t)cdesc;
    VkDescSlot *d = MQ.desc + MQ.ndesc;
    memset(d, 0, total * sizeof *d);   // unused bytes zero: equal descriptors compare equal
    for (uint32_t i = 0, nd = 0; i < info->n_bindings; i++) {
        const VkPipeBinding *b = &info->bindings[i];
        const GfxShaderRes *r = b->res;
        uint32_t cnt = b->count ? b->count : 1;
        VkDescSlot *o = &d[nd];
        int st = nvn_stage_of(b->stage);
        if (b->kind == VK_PIPE_SUPPORT) {
            float in[3];
            support_in(ex, in);
            if (!MQ.sup.buffer || memcmp(in, MQ.sup_in, sizeof in)) {
                VkBuffer sb;
                VkDeviceSize so;
                support_fill(scratch(ex, SUPPORT_BYTES, &sb, &so), in[0], in[1], in[2]);
                MQ.sup = (VkDescriptorBufferInfo){sb, so, SUPPORT_BYTES};
                memcpy(MQ.sup_in, in, sizeof in);
            }
            o->buffer = MQ.sup;
        } else if (b->kind == GFX_RES_UBO) {
            for (uint32_t k = 0; k < cnt; k++) o[k].buffer = ubo_info(ex, st, r->slot + (int)k);
        } else if (b->kind == GFX_RES_SSBO && !(r->flags & 1)) {   // read only: instance data of foliage and such
            for (uint32_t k = 0; k < cnt; k++) o[k].buffer = ssbo_info(ex, st, r, 0);
        } else if (b->kind == GFX_RES_TEX && b->type != VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER) {
            vk_texture_lock();
            for (uint32_t k = 0; k < cnt; k++) {
                int slot = r->slot + (int)k;
                tex_info(ex, r, slot >= 0 && slot < 128 ? ex->tex[st][slot] : 0, 0, r->separate, &o[k].image);
            }
            vk_texture_unlock();
        } else {
            return 0;
        }
        nd += cnt;
    }
    uint32_t n_sets = info->n_sets ? info->n_sets : 1;
    q->set[0] = q->set[1] = VK_NULL_HANDLE;
    for (uint32_t s = 0, i0 = 0, k0 = 0; s < n_sets && s < 2; s++) {
        uint32_t i1 = i0, k1 = k0;
        while (i1 < info->n_bindings && (n_sets == 1 || info->bindings[i1].set == s))
            k1 += info->bindings[i1].count ? info->bindings[i1].count : 1, i1++;
        if (i1 > i0 && !(s ? info->push1 : info->push)) {
            VkDescriptorSetLayout sl = s ? info->set_layout1 : info->set_layout;
            if (MQ.aset[s] && MQ.aset_layout[s] == sl && MQ.aset_n[s] == k1 - k0 &&
                !memcmp(MQ.aset_d[s], d + k0, (k1 - k0) * sizeof *d)) {
                q->set[s] = MQ.aset[s];
                MO.e_setreuse++;
            } else {
                static VkWriteDescriptorSet w[MAX_BINDINGS];
                VkDescriptorSet set = alloc_set(ex, sl);
                for (uint32_t i = i0, k = k0; i < i1; k += info->bindings[i].count ? info->bindings[i].count : 1, i++)
                    w[i - i0] = mo_write(&info->bindings[i], d + k), w[i - i0].dstSet = set;
                vkUpdateDescriptorSets(vk_ctx()->device, i1 - i0, w, 0, NULL);
                q->set[s] = MQ.aset[s] = set, MQ.aset_layout[s] = sl, MQ.aset_n[s] = k1 - k0;
                memcpy(MQ.aset_d[s], d + k0, (k1 - k0) * sizeof *d);
                MO.e_setw++;
            }
        }
        i0 = i1, k0 = k1;
    }
    q->desc = MQ.ndesc;
    MQ.ndesc += total;
    return 1;
}

// The draw's vertex buffers as bind_vertex_buffers resolves them, into MQ.vb; the previous entry again while the
// ranges and vk_mem_gen are the same (most motion draws in a row are one model's parts).
static uint32_t mo_vertex_buffers(VkExec *ex) {
    uint64_t gen = vk_mem_gen();
    if (MQ.nvb && MQ.vgen == gen && !memcmp(MQ.vin, ex->vbuf, sizeof MQ.vin)) {
        const MoVb *l = &MQ.vb[MQ.nvb - 1];
        int same = 1;
        if (!vk_pipeline_vertex_input_dynamic())
            for (int b = 0; b < 16 && same; b++) same = l->str[b] == ex->pipe.vstream[b].stride;
        if (same) return (uint32_t)MQ.nvb - 1;
    }
    MQ.vb = mo_grow(MQ.vb, &MQ.cvb, MQ.nvb + 1, sizeof *MQ.vb);
    MoVb *v = &MQ.vb[MQ.nvb];
    for (int b = 0; b < 17; b++) {
        v->b[b] = Z.zbuf, v->o[b] = 0, v->sz[b] = ZERO_BYTES, v->str[b] = b < 16 ? ex->pipe.vstream[b].stride : 0;
        if (b == VK_PIPE_ZERO_BINDING) continue;
        const NVNbufferRange *r = &ex->vbuf[b];
        VkMemSpan sp;
        if (!r->address || !vk_mem_resolve(r->address, 1, &sp)) continue;
        VK_MEM_USE(sp);
        v->b[b] = sp.buffer, v->o[b] = sp.offset, v->sz[b] = r->size && r->size < sp.avail ? r->size : sp.avail;
    }
    memcpy(MQ.vin, ex->vbuf, sizeof MQ.vin);
    MQ.vgen = gen;
    return (uint32_t)MQ.nvb++;
}

// Records the list into cb (the replay's command buffer, inside the motion pass's rendering), skipping what the draw
// before already set.
static void mo_emit(VkCommandBuffer cb) {
    int dyn_vi = vk_pipeline_vertex_input_dynamic();
    VkPipeline pipe = VK_NULL_HANDLE;
    uint32_t st = UINT32_MAX, vb = UINT32_MAX;
    VkPipelineLayout lay[2] = {0};
    const VkPipeInfo *pinfo[2] = {0};
    int pdesc[2] = {-1, -1};      // the pushed set's descriptors (MQ.desc index) under lay/pinfo, -1 none
    VkDescriptorSet bound[2] = {0};
    const MoDraw *p = NULL;
    for (int n = 0; n < MQ.n; p = &MQ.d[n++]) {
        const MoDraw *q = &MQ.d[n];
        const VkPipeInfo *info = q->info;
        if (q->pipe != pipe) vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, q->pipe), pipe = q->pipe, MO.e_pipe++;
        if (q->st != st) vk_pipeline_set_dynamic(cb, &MQ.st[q->st]), st = q->st, MO.e_dyn++;
        if (!p || memcmp(&q->v, &p->v, sizeof q->v)) vkCmdSetViewportWithCount(cb, 1, &q->v);
        if (!p || memcmp(&q->sc, &p->sc, sizeof q->sc)) vkCmdSetScissorWithCount(cb, 1, &q->sc);
        if (!p || memcmp(q->bias, p->bias, sizeof q->bias)) vkCmdSetDepthBias(cb, q->bias[0], q->bias[1], q->bias[2]);
        uint32_t n_sets = info->n_sets ? info->n_sets : 1;
        const VkDescSlot *d = MQ.desc + q->desc;
        for (uint32_t s = 0, i0 = 0, k0 = 0; s < n_sets && s < 2; s++) {
            uint32_t i1 = i0, k1 = k0;
            while (i1 < info->n_bindings && (n_sets == 1 || info->bindings[i1].set == s))
                k1 += info->bindings[i1].count ? info->bindings[i1].count : 1, i1++;
            if (i1 > i0) {
                if (s ? info->push1 : info->push) {
                    // the bindings that differ from the last push of this layout and pipeline (all after another)
                    int same = lay[s] == info->layout && pinfo[s] == info && pdesc[s] >= 0;
                    const VkDescSlot *pd = same ? MQ.desc + pdesc[s] : NULL;
                    VkWriteDescriptorSet w[MAX_BINDINGS];
                    uint32_t nw = 0;
                    for (uint32_t i = i0, k = k0; i < i1; i++) {
                        uint32_t cnt = info->bindings[i].count ? info->bindings[i].count : 1;
                        if (!same || memcmp(d + k, pd + k, cnt * sizeof *d)) w[nw++] = mo_write(&info->bindings[i], d + k);
                        k += cnt;
                    }
                    if (nw) rec_push(vk_pipeline_push_fn(), cb, VK_PIPELINE_BIND_POINT_GRAPHICS, info->layout, s, nw, w);
                    MO.e_push += nw != 0, MO.e_pushw += nw;
                    lay[s] = info->layout, pinfo[s] = info, pdesc[s] = (int)q->desc, bound[s] = VK_NULL_HANDLE;
                } else if (q->set[s] != bound[s] || lay[s] != info->layout) {
                    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, info->layout, s, 1, &q->set[s], 0, NULL);
                    bound[s] = q->set[s], lay[s] = info->layout, pinfo[s] = info, pdesc[s] = -1, MO.e_sets++;
                }
            }
            i0 = i1, k0 = k1;
        }
        if (q->vb != vb) {
            const MoVb *v = &MQ.vb[q->vb];
            vkCmdBindVertexBuffers2(cb, 0, 17, v->b, v->o, v->sz, dyn_vi ? NULL : v->str);
            vb = q->vb, MO.e_vb++;
        }
        if (!p || q->ib != p->ib || q->ib_off != p->ib_off || q->it != p->it) vkCmdBindIndexBuffer(cb, q->ib, q->ib_off, q->it);
        vkCmdDrawIndexed(cb, q->count, q->instances, 0, q->base_vertex, q->base_instance);
        MO.e_draws++;
    }
}

static void mvobj_draw_(VkExec *ex, const p_draw_idx *d, VkBuffer ib, VkDeviceSize ib_off, uint64_t t0);
static void mvobj_draw(VkExec *ex, const p_draw_idx *d, VkBuffer ib, VkDeviceSize ib_off) {
    if (MO.failed || !vk_mvprev_on() || !B.st_valid) return;
    static int timed = -1;
    if (timed < 0) timed = rt_knob("MVOBJ_TIMES", "0");   // NX2NAVI_MVOBJ_TIMES=1: the summary's times (the clock reads cost)
    if (!timed) return mvobj_draw_(ex, d, ib, ib_off, 0);
    uint64_t t0 = kern_now_ns();
    mvobj_draw_(ex, d, ib, ib_off, t0);
    MO.ns_check += kern_now_ns() - t0;
}

// NX2NAVI_MVOBJ_BANKS=1 (test): for scene draws, the words of each other uniform bank (c3, c6, c8-c16) that changed
// since the same shader's draw last frame, logged at frame 3000: where the time that moves wind and water lives
static void mvobj_banks(VkExec *ex) {
    static int on = -1;
    if (on < 0) on = rt_knob("MVOBJ_BANKS", "0");
    if (!on || !B.st_valid) return;
    uint64_t frame = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    if (frame < 2999 || frame > 3000) return;
    static struct { const GfxShader *vs; int bank; uint64_t frame; float w[64]; int n; } tab[4096];
    static int logged;
    const GfxShaderRes *ur;
    int nu = gfx_shader_resources(B.st.vs, GFX_RES_UBO, &ur);
    for (int i = 0; i < nu; i++) {
        int bank = ur[i].slot;
        if (bank < 3 || bank == 4 || bank == 5 || bank == 7 || bank > 16) continue;
        const p_bind_buf *u = &ex->ubo[NVN_STAGE_VS][bank - 3];
        uint32_t n = u->size < 256 ? (uint32_t)u->size / 4 : 64;
        const float *now = n ? (const float *)vk_usnap_bytes(u->addr, n * 4) : NULL;
        if (!now) continue;
        uint32_t k = (uint32_t)(((uintptr_t)B.st.vs ^ (uint64_t)bank * 0x9E3779B97F4A7C15ull) * 0xff51afd7ed558ccdull >> 52);
        if (frame == 2999) {
            tab[k].vs = B.st.vs, tab[k].bank = bank, tab[k].frame = frame, tab[k].n = (int)n;
            memcpy(tab[k].w, now, n * 4);
        } else if (tab[k].vs == B.st.vs && tab[k].bank == bank && tab[k].frame == 2999 && logged < 80) {
            char line[700];
            int o = 0, nd = 0;
            for (int j = 0; j < tab[k].n && j < (int)n && o < 600; j++)
                if (memcmp(&tab[k].w[j], &now[j], 4)) nd++, o += snprintf(line + o, sizeof line - o, " [%d] %g>%g", j, tab[k].w[j], now[j]);
            if (nd) logged++, rt_log("mvbanks: vs %p c%d size %#llx: %d words%s", (const void *)B.st.vs, bank,
                                     (unsigned long long)u->size, nd, line);
            tab[k].frame = 0;
        }
    }
}

// 1: old is cur a little changed or the same (every float within 10 %, at least 1.0 and at most 4.0); *changed: not
// the same. The cap: NX2navi's world positions are in the thousands, and 10 % of them let another object's block a few
// units away pass as this one moved; nothing moves 4 units in a frame.
static int mo_similar(const uint8_t *cur, const uint8_t *old, uint64_t n, int *changed) {
    *changed = 0;
    if (!memcmp(cur, old, n)) return 1;
    // word by word as integers; float arithmetic only on two normal floats (the blocks hold integers too, and
    // denormal arithmetic took this loop 1.9 million cycles a block)
    const uint32_t *x = (const uint32_t *)cur, *y = (const uint32_t *)old;
    for (uint64_t w = 0; w < n / 4; w++) {
        if (x[w] == y[w]) continue;
        uint32_t ex = x[w] >> 23 & 0xff, ey = y[w] >> 23 & 0xff;
        if (!ex || ex == 0xff || !ey || ey == 0xff) continue;   // not both normal floats: an integer or a tiny value
        float fx, fy;
        memcpy(&fx, &x[w], 4), memcpy(&fy, &y[w], 4);
        float dd = fabsf(fx - fy), tol = 0.1f * fabsf(fy);
        if (dd > (tol < 1.0f ? 1.0f : tol > 4.0f ? 4.0f : tol)) return 0;
    }
    *changed = 1;
    return 1;
}

// Last frame's copy of a block, accepted when it is this frame's block a little changed (mo_similar,
// which see): a block of another object the address rule found would make motion up. 1 and *out: found (the block itself
// when it is unchanged, *changed 0); 0: none, the draw keeps this frame's block there.
static int mo_prev_block(int index, const p_bind_buf *u, VkDescriptorBufferInfo *out, int *changed) {
    uint64_t max = LIM.ubo_max, n = u->size < max ? u->size : max, pa;
    int how;
    *changed = 0;
    // c9 and c11 keep this frame's block. With the copies the address rule finds for c11 and last frame's c9, the
    // c4 c7 c9 c11 draws (walls, cliffs, arches) got several pixels of motion under a still camera (NX2NAVI_MV_CHECK,
    // those draws alone: 34 % of the picture moving; with c11 kept, 0.8 %). With c11 kept, their c9 still changes every
    // frame while nothing moves (the check's per-draw line: about 120 such draws a frame changed in c9 alone), the
    // object scale the picture shows is 0.6-0.8 of theirs, and with Link stopped against a cliff 40-77 % of the picture
    // got 1-3 px of motion that flipped direction, three times the warp error of the camera's vectors. With c9 kept too,
    // the turning run is as good (object warp error 0.46 of the camera's). Why their copies are wrong is not known.
    // NX2NAVI_MVPREV_SKIP=<hex bank mask> (test) replaces the default a00 (bit n: cn).
    static long skip = -2;
    if (skip == -2) {
        const char *e = cfg_get("NX2NAVI_MVPREV_SKIP");
        skip = e && *e ? strtol(e, NULL, 16) : 1L << 9 | 1L << 11;
    }
    if (skip >> (index + 3) & 1) return 0;
    uint64_t c0 = __builtin_ia32_rdtsc();
    int found = vk_mvprev_find(index, u->addr, u->size, &pa, &how);
    uint64_t c1 = __builtin_ia32_rdtsc();
    MO.cyc[0] += c1 - c0, MO.ncyc++, MO.big += n;
    if (index >= 0 && index < 14) MO.lk[index][found ? how : MVPREV_NONE]++;
    if (!found || !vk_usnap_find(pa, n, LIM.ubo_align, out)) return 0;
    if (how == MVPREV_SELF) return 1;   // bound at the same address last frame: not written again
    const uint8_t *map = vk_usnap_map(), *cur = vk_usnap_bytes(u->addr, n);
    uint64_t c2 = __builtin_ia32_rdtsc();
    MO.cyc[1] += c2 - c1;
    if (!cur) return 0;
    const uint8_t *old = map + out->offset;
    static long trust = -2;   // NX2NAVI_MVPREV_TRUST=<hex bank mask> (test): the copy found is taken without the check
    if (trust == -2) {
        const char *e = cfg_get("NX2NAVI_MVPREV_TRUST");
        trust = e && *e ? strtol(e, NULL, 16) : 0;
    }
    if (trust >> (index + 3) & 1) {
        *changed = memcmp(cur, old, n) != 0;
        return 1;
    }
    int r = mo_similar(cur, old, n, changed);
    MO.cyc[2] += __builtin_ia32_rdtsc() - c2;
    if (!r && index >= 0 && index < 14) MO.lk[index][4]++;
    return r;
}

// mo_prev_block once per block and list: many draws share their blocks (the scene's c14, a model's c7 and c9 across
// its parts), and the lookup with its comparison was most of the checks' time.
static int mo_prev_cached(int index, const p_bind_buf *u, VkDescriptorBufferInfo *out, int *changed) {
    static struct { uint64_t gen, addr, size; int index, r, changed; VkDescriptorBufferInfo out; } bc[4096];
    uint32_t i = (uint32_t)((u->addr ^ (uint64_t)index << 56 ^ u->size << 40) * 0x9E3779B97F4A7C15ull >> 52);
    if (bc[i].gen == mo_gen && bc[i].addr == u->addr && bc[i].size == u->size && bc[i].index == index) {
        *out = bc[i].out, *changed = bc[i].changed;
        return bc[i].r;
    }
    int r = mo_prev_block(index, u, out, changed);
    bc[i].gen = mo_gen, bc[i].addr = u->addr, bc[i].size = u->size, bc[i].index = index;
    bc[i].r = r, bc[i].changed = *changed, bc[i].out = *out;
    return r;
}

static void mvobj_draw_(VkExec *ex, const p_draw_idx *d, VkBuffer ib, VkDeviceSize ib_off, uint64_t t0) {
    mvobj_banks(ex);
    const VkExecTarget *z = &ex->depth;
    if (!z->image || z->width < DLAA_SCENE_MIN_W || z->format != VK_FORMAT_D32_SFLOAT || B.st.samples > 1 || B.st.raster_discard)
        return;
    // solid geometry only, which tests and writes the scene depth: effects, particles and screen quads write none, and
    // drawn again they would put their motion on the surfaces behind them (DLSS takes those from the surface)
    if (!B.st.ds.dt || !B.st.ds.dw) return;
    // a vertex shader that reads a user bank besides the camera's (c4) may move on its own
    const GfxShaderRes *ur;
    int nu = gfx_shader_resources(B.st.vs, GFX_RES_UBO, &ur), reads = 0;
    for (int i = 0; i < nu; i++) reads |= ur[i].slot >= 3 && ur[i].slot <= 16 && ur[i].slot != 4;
    if (!reads) return;
    MO.cand++;
    uint64_t tp = t0 ? kern_now_ns() : 0;
    if (t0) MO.ns_part[0] += tp - t0, MO.n_part[0]++;
    uint64_t frame = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    if (MQ.frame != frame || MQ.fence != ex->fence || MQ.depth_fmt != z->format) mo_begin(ex, frame);
    if (!__atomic_load_n(&LIM.ubo_max, __ATOMIC_ACQUIRE)) limits_init();
    // the variant: the translation of the same code and answers, kept per shader once found; it names the banks the
    // position depends on (0: none, the shader is skipped from then on)
    static struct { const GfxShader *of, *vs; uint32_t banks; } vmemo[1024];
    uint32_t vi = (uint32_t)((uintptr_t)B.st.vs * 0x9E3779B97F4A7C15ull >> 54);
    if (vmemo[vi].of != B.st.vs) {
        int has, pending = 0;
        sf_mvprev = 1;
        const GfxShader *v = shader_for(ex, NVN_STAGE_VS, &has, &pending, 0);
        sf_mvprev = 0;
        if (!v || pending) {
            MO.building++;
            return;
        }
        GfxShaderIo io[32];
        uint32_t extra = gfx_shader_outputs(v, io) & ~gfx_shader_outputs(B.st.vs, io);
        uint32_t banks = extra ? gfx_shader_mvprev_banks(v) & 0x1fff8u : 0;
        vmemo[vi].of = B.st.vs, vmemo[vi].vs = v, vmemo[vi].banks = banks;
        vk_mvprev_want((banks & ~(1u << 4)) >> 3);   // binding index = bank - 3
    }
    const GfxShader *vs = vmemo[vi].vs;
    uint32_t banks = vmemo[vi].banks;
    if (!banks || !(banks & ~(1u << 4))) {
        MO.no_out++;
        return;
    }
    // a position without the scene camera (c4) has a view of its own (c8 c9 c10 [c13]: its c8 is at a new address every
    // frame, last frame's is never found): drawn again it would get this frame's view and no camera motion
    // (NX2NAVI_MV_CHECK while turning: 6.2 px end-point error against the camera vectors' 1.7)
    static int noc4 = -1;
    if (noc4 < 0) noc4 = rt_knob("MVOBJ_NOC4", "0");   // NX2NAVI_MVOBJ_NOC4=1 (test): drawn again all the same
    if (!(banks >> 4 & 1) && !noc4) {
        MO.no_out++;
        return;
    }
    // only positions that read an object's own blocks: bones (c5), place (c7) or c6. The rest (terrain and grass: c14
    // c15 c16, with or without c9) measured no better than the camera's motion (NX2NAVI_MV_CHECK, turning, those draws
    // alone: object warp error 0.112 against the camera's 0.113, the object part 0.4 of what the picture shows) and
    // were half the draws checked and listed. NX2NAVI_MVOBJ_ALL=1 (test) takes them again.
    static int all = -1;
    if (all < 0) all = rt_knob("MVOBJ_ALL", "0");
    if (!all && !(banks & (1u << 5 | 1u << 6 | 1u << 7))) {
        MO.no_out++;
        return;
    }
    // NX2NAVI_MVOBJ_ONLY=<hex bank mask> (test): only draws whose position reads no bank outside the mask (bit n: cn)
    static long only = -2;
    if (only == -2) {
        const char *e = cfg_get("NX2NAVI_MVOBJ_ONLY");
        only = e && *e ? strtol(e, NULL, 16) : -1;
    }
    if (only >= 0 && banks & ~(uint32_t)only & ~(1u << 4)) return;
    static long exact = -2;   // NX2NAVI_MVOBJ_EXACT=<hex bank mask> (test): only draws with exactly these position banks
    if (exact == -2) {
        const char *e = cfg_get("NX2NAVI_MVOBJ_EXACT");
        exact = e && *e ? strtol(e, NULL, 16) : -1;
    }
    if (exact >= 0 && banks != (uint32_t)exact) return;
    mo_mask_note(banks, B.st.vs);
    // last frame's camera and blocks; held still when none of the position's blocks changed (most of the scenery)
    VkDescriptorBufferInfo cam, prevb[17];
    float jit[2];
    if (!vk_dlaa_prev_camera(&cam, jit)) {
        MO.nodata++, MO.why[0]++;
        return;
    }
    static struct { uint64_t frame, key; int moving; VkDescriptorBufferInfo b[17]; uint32_t have; } memo[8192];
    uint64_t key = banks;
    for (int b = 3; b <= 16; b++)
        if (banks >> b & 1 && b != 4) key = (key ^ ex->ubo[NVN_STAGE_VS][b - 3].addr) * 0x100000001b3ull;
    uint32_t mi = (uint32_t)(key * 0xff51afd7ed558ccdull >> 51);
    uint32_t have = 0;
    if (memo[mi].frame == frame && memo[mi].key == key) {
        if (!memo[mi].moving) {
            if (t0) MO.ns_still += kern_now_ns() - tp;
            return (void)MO.still++;
        }
        memcpy(prevb, memo[mi].b, sizeof prevb), have = memo[mi].have;
    } else {
        MO.misses++;
        int moving = 0;
        for (int b = 3; b <= 16; b++) {
            if (!(banks >> b & 1) || b == 4) continue;
            const p_bind_buf *u = &ex->ubo[NVN_STAGE_VS][b - 3];
            int changed;
            if (u->addr && u->size && mo_prev_cached(b - 3, u, &prevb[b], &changed))
                have |= 1u << b, moving |= changed << b;   // the banks whose block changed
        }
        memo[mi].frame = frame, memo[mi].key = key, memo[mi].moving = moving;
        memcpy(memo[mi].b, prevb, sizeof prevb), memo[mi].have = have;
        if (!moving) {
            if (t0) MO.ns_still += kern_now_ns() - tp;
            return (void)MO.still++;
        }
    }
    if (t0) MO.ns_part[1] += kern_now_ns() - tp, MO.n_part[1]++, tp = kern_now_ns();
    GfxShaderIo io[32];
    uint32_t extra = gfx_shader_outputs(vs, io) & ~gfx_shader_outputs(B.st.vs, io);
    const GfxShaderRes *r;
    if (gfx_shader_resources(vs, GFX_RES_IMG, &r)) {
        MO.storage++;
        return;
    }
    const GfxShader *fs = mo_fs(31 - __builtin_clz(extra));
    if (!fs) {
        MO.no_fs++;
        return;
    }
    VkPipeState ms = B.st;
    ms.vs = vs, ms.fs = fs, ms.feedback = 0;
    memset(ms.color_fmt, 0, sizeof ms.color_fmt);
    ms.color_fmt[0] = NVN_FORMAT_RG16F;
    // blending off (blend_en below); the factors stay the game's, valid NVN values
    ms.color.blend_en = 0, ms.color.logic_op = 0;
    ms.chmask.mask = 0x3;   // r and g of target 0
    ms.ms.a2c = 0;
    // the game's own test with its equal case (LESS -> LEQUAL, GREATER -> GEQUAL): the surface the draw left in the
    // scene depth passes, anything nearer hides it (NX2NAVI_MVOBJ_EQUAL=1: EQUAL, for comparison)
    static int eq = -1;
    if (eq < 0) eq = rt_knob("MVOBJ_EQUAL", "0");
    int df = B.st.ds.df;
    df = eq ? NVN_DEPTH_FUNC_EQUAL : df == NVN_DEPTH_FUNC_LESS ? NVN_DEPTH_FUNC_LEQUAL
       : df == NVN_DEPTH_FUNC_GREATER ? NVN_DEPTH_FUNC_GEQUAL : df == NVN_DEPTH_FUNC_LEQUAL || df == NVN_DEPTH_FUNC_GEQUAL ? df
       : NVN_DEPTH_FUNC_EQUAL;
    ms.ds.dt = 1, ms.ds.dw = 0, ms.ds.df = (uint64_t)df, ms.ds.st = 0;
    // the pipeline per state, kept: the lookup hashes and locks the whole state
    static struct { VkPipeState st; VkPipeline p; const VkPipeInfo *info; int used; } pmemo[64];
    static int plast = -1;   // the entry of the last draw: a model's parts in a row share the state, no hash then
    uint32_t pi;
    if (plast >= 0 && pmemo[plast].used && !memcmp(&pmemo[plast].st, &ms, sizeof ms)) {
        pi = (uint32_t)plast;
    } else {
        uint64_t ph = 0xcbf29ce484222325ull;
        for (size_t i = 0; i < sizeof ms; i += 8) {
            uint64_t w8;
            memcpy(&w8, (const uint8_t *)&ms + i, sizeof ms - i < 8 ? sizeof ms - i : 8);
            ph = (ph ^ w8) * 0x100000001b3ull;
        }
        pi = (uint32_t)(ph >> 58);
    }
    const VkPipeInfo *info = NULL;
    VkPipeline pipe;
    int pending = 0;
    if (pmemo[pi].used && !memcmp(&pmemo[pi].st, &ms, sizeof ms)) {
        info = pmemo[pi].info;
        pipe = __atomic_load_n(&info->pipeline, __ATOMIC_ACQUIRE);   // its optimised build once swapped in
        if (!pipe) return;
    } else {
        pipe = vk_pipeline_graphics_async(&ms, &info, &pending, 0);
        if (pending || !pipe || !info) {
            MO.building += pending != 0;
            return;
        }
        pmemo[pi].st = ms, pmemo[pi].p = pipe, pmemo[pi].info = info, pmemo[pi].used = 1;
    }
    plast = (int)pi;
    uint64_t t1 = kern_now_ns();
    if (t0) MO.ns_part[2] += t1 - tp, MO.n_part[2]++;
    if (t0) MO.ns_check -= t1 - t0;   // counted as recording below
    // into the frame's list, past the primary's binding caches (mo_desc); mo_emit records it
    MQ.d = mo_grow(MQ.d, &MQ.cap, MQ.n + 1, sizeof *MQ.d);
    MoDraw *q = &MQ.d[MQ.n];
    float par_in[4] = {(float)z->width, (float)z->height, jit[0], jit[1]};
    if (!MQ.par.buffer || memcmp(par_in, MQ.par_in, sizeof par_in)) {
        VkBuffer pbuf;
        VkDeviceSize poff;
        memcpy(scratch(ex, 16, &pbuf, &poff), par_in, sizeof par_in);
        MQ.par = (VkDescriptorBufferInfo){pbuf, poff, 16};
        memcpy(MQ.par_in, par_in, sizeof par_in);
    }
    memset(mo_ovr, 0, sizeof mo_ovr);
    for (int bk = 3; bk <= 16; bk++)   // last frame's block, or this frame's where none was found
        mo_ovr[NVN_STAGE_VS][bk + 15] = bk == 4 ? cam : have >> bk & 1 ? prevb[bk] : ubo_info(ex, NVN_STAGE_VS, bk);
    mo_ovr[NVN_STAGE_FS][3] = MQ.par;
    mo_ovr_on = 1;
    int bound = mo_desc(ex, info, q);
    mo_ovr_on = 0;
    if (!bound) {
        MO.storage++;
        return;
    }
    q->pipe = pipe, q->info = info;
    if (!MQ.nst || memcmp(&MQ.st[MQ.nst - 1], &ms, sizeof ms)) {
        MQ.st = mo_grow(MQ.st, &MQ.cst, MQ.nst + 1, sizeof *MQ.st);
        MQ.st[MQ.nst++] = ms;
    }
    q->st = (uint32_t)MQ.nst - 1;
    q->v = draw_viewport(ex), q->sc = vk_exec_scissor(ex);
    q->bias[0] = ex->poly_offset.units, q->bias[1] = ex->poly_offset.clamp, q->bias[2] = ex->poly_offset.factor;
    q->vb = mo_vertex_buffers(ex);
    q->ib = ib, q->ib_off = ib_off;
    q->it = d->itype == NVN_INDEX_TYPE_UNSIGNED_SHORT ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32;
    q->count = (uint32_t)d->count, q->instances = d->instances > 0 ? (uint32_t)d->instances : 1;
    q->base_vertex = d->base_vertex, q->base_instance = (uint32_t)d->base_instance;
    q->cam = ex->ubo[NVN_STAGE_VS][1].addr;
    MQ.n++;
    MO.draws++;
    for (int i = 0; i < 16; i++)   // the frame's draws per position banks and changed banks (vk_mvobj_frame_masks)
        if (!MF[i].n || (MF[i].mask == banks && MF[i].chg == (uint32_t)memo[mi].moving)) {
            MF[i].mask = banks, MF[i].chg = (uint32_t)memo[mi].moving, MF[i].n++;
            break;
        }
    if (t0) MO.ns_rec += kern_now_ns() - t1, MO.n_rec++;
}

int vk_mvobj_execute(VkExec *ex, uint64_t cam, VkImageView mv, VkImageView depth, VkFormat depth_fmt, uint32_t w,
                     uint32_t h) {
    static uint64_t frames;
    if (++frames % 60 == 0)
        rt_log("vk_exec: per-object motion: %.2f ms a frame checking, %.2f ms listing (%.1f us a draw), %.2f ms recording",
               MO.ns_check / 1e6 / (frames ? frames : 1), MO.ns_rec / 1e6 / (frames ? frames : 1),
               MO.n_rec ? MO.ns_rec / 1e3 / MO.n_rec : 0.0, MO.ns_emit / 1e6 / (frames ? frames : 1));
    if (frames % 600 == 0) {
        char l[1400];
        int o = 0;
        for (int i = 0; i < 32 && MM[i].n && o < 1300; i++)
            o += snprintf(l + o, sizeof l - o, " %#x:%llu (u%d s%d t%d)", MM[i].mask, (unsigned long long)MM[i].n, MM[i].ubo,
                          MM[i].ssbo, MM[i].tex);
        if (o) rt_log("vk_exec: per-object motion position banks (bit n: cn) and draws:%s", l);
        o = 0;
        for (int i = 0; i < 14 && o < 1300; i++)
            if (MO.lk[i][0] + MO.lk[i][1] + MO.lk[i][2] + MO.lk[i][3])
                o += snprintf(l + o, sizeof l - o, " c%d %llu/%llu/%llu/%llu/%llu", i + 3, (unsigned long long)MO.lk[i][0],
                              (unsigned long long)MO.lk[i][1], (unsigned long long)MO.lk[i][2],
                              (unsigned long long)MO.lk[i][3], (unsigned long long)MO.lk[i][4]);
        if (o) rt_log("vk_exec: per-object motion last-frame lookups (none/self/side/nearer/refused):%s", l);
    }
    if (frames % 60 == 0 && MO.ran)
        rt_log("vk_exec: per-object motion recorded a frame: %.0f draws, %.0f pipeline binds, %.0f dynamic states, %.0f "
               "pushes of %.1f bindings, %.0f set binds, %.0f vertex buffer binds; sets written %.0f, reused %.0f",
               (double)MO.e_draws / MO.ran, (double)MO.e_pipe / MO.ran, (double)MO.e_dyn / MO.ran, (double)MO.e_push / MO.ran,
               MO.e_push ? (double)MO.e_pushw / MO.e_push : 0.0, (double)MO.e_sets / MO.ran, (double)MO.e_vb / MO.ran,
               (double)MO.e_setw / MO.ran, (double)MO.e_setreuse / MO.ran);
    if (frames % 60 == 0)
        rt_log("vk_exec: per-object motion checks: filter %.2f ms (%.0f a frame), data %.2f ms (%.0f), variant and "
               "pipeline %.2f ms (%.0f); %.0f objects looked up; still draws %.2f ms", MO.ns_part[0] / 1e6 / frames,
               (double)MO.n_part[0] / frames,
               MO.ns_part[1] / 1e6 / frames, (double)MO.n_part[1] / frames, MO.ns_part[2] / 1e6 / frames,
               (double)MO.n_part[2] / frames, (double)MO.misses / frames, MO.ns_still / 1e6 / frames);
    if (frames % 60 == 0 && MO.ncyc)
        rt_log("vk_exec: per-object motion lookups: %llu cycles finding last frame's, %llu the copies, %llu comparing, "
               "a lookup; %llu bytes a lookup", (unsigned long long)(MO.cyc[0] / MO.ncyc),
               (unsigned long long)(MO.cyc[1] / MO.ncyc), (unsigned long long)(MO.cyc[2] / MO.ncyc),
               (unsigned long long)(MO.big / MO.ncyc));
    if (frames % 60 == 0)
        rt_log("vk_exec: per-object motion: %llu draws read place or bones, %llu of them still, %llu drawn again, %llu without last frame's data, %llu "
               "waiting for a shader or pipeline; %llu frames ran, %llu dropped (no data: camera %llu, bones %llu, place %llu, "
               "bones copy %llu, place copy %llu; uniform alignment %llu; variant without its output %llu, with storage %llu, "
               "no fragment shader %llu; another camera %llu)",
               (unsigned long long)MO.cand, (unsigned long long)MO.still, (unsigned long long)MO.draws, (unsigned long long)MO.nodata,
               (unsigned long long)MO.building,
               (unsigned long long)MO.ran, (unsigned long long)MO.dropped, (unsigned long long)MO.why[0],
               (unsigned long long)MO.why[1], (unsigned long long)MO.why[2], (unsigned long long)MO.why[3],
               (unsigned long long)MO.why[4], (unsigned long long)LIM.ubo_align, (unsigned long long)MO.no_out,
               (unsigned long long)MO.storage, (unsigned long long)MO.no_fs, (unsigned long long)MO.other_cam);
    if (!MQ.n) return 0;
    static int exec = -1;
    if (exec < 0) exec = rt_knob("MVOBJ_EXEC", "1");   // NX2NAVI_MVOBJ_EXEC=0: listed, never recorded (test)
    if (!exec) {
        mo_drop();
        return 0;
    }
    uint64_t frame = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    if (MQ.frame != frame || MQ.fence != ex->fence || MQ.depth_fmt != depth_fmt) {
        mo_drop();
        return 0;
    }
    int fo = 0;
    for (int i = 0; i < 16 && MF[i].n && fo < (int)sizeof fm_last - 40; i++)
        fo += snprintf(fm_last + fo, sizeof fm_last - fo, " %#x/%#x:%u", MF[i].mask, MF[i].chg, MF[i].n);
    if (!fo) fm_last[0] = 0;
    // only draws with the scene camera block (vk_dlaa.c mv_pass's): last frame's camera they get is that block's, and
    // a draw of another view would move by the difference between the two views
    int kept = 0;
    for (int i = 0; i < MQ.n; i++)
        if (MQ.d[i].cam == cam) MQ.d[kept++] = MQ.d[i];
        else MO.other_cam++;
    MQ.n = kept;
    if (!MQ.n) {
        mo_drop();
        return 0;
    }
    uint64_t t0 = kern_now_ns();
    VkCommandBuffer cb = ex->cb;
    VkRenderingAttachmentInfo ca = {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = mv,
                                    .imageLayout = VK_IMAGE_LAYOUT_GENERAL, .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
                                    .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
    VkRenderingAttachmentInfo da = {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = depth,
                                    .imageLayout = VK_IMAGE_LAYOUT_GENERAL, .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
                                    .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
    VkRenderingInfo ri = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{0, 0}, {w, h}}, .layerCount = 1,
                          .colorAttachmentCount = 1, .pColorAttachments = &ca, .pDepthAttachment = &da};
    vkCmdBeginRendering(cb, &ri);   // through the recorder (vk_rec.h), in order with the draws and the end
    mo_emit(cb);
    vkCmdEndRendering(cb);
    // the draws left their pipeline, descriptors, buffers and dynamic state bound: the next game draw sets everything
    vk_exec_draw_new_cb();
    ex->dirty = VK_EXEC_DIRTY_ALL;
    ex->recorded += MQ.n;
    MQ.n = MQ.nst = MQ.nvb = 0, MQ.ndesc = 0, MQ.frame = 0;
    MO.ran++;
    MO.ns_emit += kern_now_ns() - t0;
    return 1;
}

// NVN2 mesh draws (DrawMeshTasks): count task workgroups, or mesh workgroups for a program without a task stage, through
// VK_EXT_mesh_shader. The stages are the translator's TaskEXT / MeshEXT modules (tools/xlate/MeshSpirv.cs).
static PFN_vkCmdDrawMeshTasksEXT draw_mesh_fn(void) {
    static PFN_vkCmdDrawMeshTasksEXT fn;
    static int looked;
    if (!looked) {
        looked = 1;
        if (vk_ctx()->has_mesh_shader) fn = (PFN_vkCmdDrawMeshTasksEXT)vkGetDeviceProcAddr(vk_ctx()->device, "vkCmdDrawMeshTasksEXT");
    }
    return fn;
}

static void op_draw_mesh(VkExec *ex, const p_draw_mesh *d) {
    PFN_vkCmdDrawMeshTasksEXT fn = draw_mesh_fn();
    if (!fn) return warn_once(W_MESH, "mesh draws need VK_EXT_mesh_shader: skipped");
    static int on = -1;
    if (on < 0) on = rt_knob("MESH", "1");   // NX2NAVI_MESH=0: mesh draws are skipped (diagnosis)
    if (!d->count || !on) return;
    c0_binst = 0;
    if (!prepare_draw_ex(ex, NVN_DRAW_PRIMITIVE_TRIANGLES, 0x7fffffff, 0, 1)) return;
    if (d->first) warn_once(W_MESH, "a mesh draw from workgroup %u: drawn from workgroup 0", d->first);
    vk_exec_frame.draws++;
    static uint32_t n_logged;
    if (n_logged < 8 && knob_log_mesh()) {
        n_logged++;
        rt_log("vk: mesh draw: %u workgroups, program %llu", d->count, (unsigned long long)ex->prog->serial);
    }
    rec_draw_mesh(fn, ex->cb, d->count, 1, 1);
    ex->recorded++;
}

// The record NVN's DrawMeshTasksIndirect reads is {count, first} (as NV_mesh_shader's), and Vulkan's {x, y, z}: read as
// it is, y = first (0) drew nothing. Each draw gets an entry of this ring, {count, 1, 1}: y and z filled with 1 and x
// copied from the game's record on the GPU, between the pass's end and the draw (the record is written by the culling
// compute earlier in the frame and never comes back to guest memory). `first` is not applied: the game leaves it 0.
// MESH_RING entries outlast the frames in flight: an entry is rewritten only after a barrier behind its last draw.
enum { MESH_RING = 4096 };
static VkBuffer mesh_ring;
static uint32_t mesh_ring_next;

static int mesh_ring_entry(VkExec *ex, const VkMemSpan *src, VkDeviceSize *at) {
    if (!mesh_ring) {
        VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = 12ull * MESH_RING,
                                 .usage = VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        VmaAllocation a;
        if (vmaCreateBuffer(vk_ctx()->vma, &bi, &ai, &mesh_ring, &a, NULL) != VK_SUCCESS) {
            mesh_ring = VK_NULL_HANDLE;
            return 0;
        }
    }
    *at = 12ull * (mesh_ring_next++ % MESH_RING);
    vk_exec_barrier(ex);   // ends the pass (the next draw opens it again) and orders the culling's writes before the copy
    vkCmdFillBuffer(ex->cb, mesh_ring, *at + 4, 8, 1);
    VkBufferCopy cp = {src->offset, *at, 4};
    vkCmdCopyBuffer(ex->cb, src->buffer, mesh_ring, 1, &cp);
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                           .srcStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                           .dstStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
                           .dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(ex->cb, &di);
    ex->recorded += 3;
    return 1;
}

// DrawMeshTasksIndirect: one draw from its ring entry (mesh_ring_entry); x2-x4 of the call vary like leftovers, so they
// are not read as a count or stride.
static void op_draw_mesh_ind(VkExec *ex, const p_draw_mesh_ind *d) {
    static PFN_vkCmdDrawMeshTasksIndirectEXT fn;
    static int looked;
    if (!looked) {
        looked = 1;
        if (vk_ctx()->has_mesh_shader)
            fn = (PFN_vkCmdDrawMeshTasksIndirectEXT)vkGetDeviceProcAddr(vk_ctx()->device, "vkCmdDrawMeshTasksIndirectEXT");
    }
    if (!fn) return warn_once(W_MESH, "mesh draws need VK_EXT_mesh_shader: skipped");
    static int on = -1;
    if (on < 0) on = rt_knob("MESH", "1");
    if (!on) return;
    VkMemSpan ip;
    if (!vk_mem_resolve(d->ind, 12, &ip) || ip.offset % 4)
        return warn_once(W_INDIRECT, "indirect mesh draw arguments at %#llx are not backed or not aligned: skipped",
                         (unsigned long long)d->ind);
    c0_binst = 0;
    VK_MEM_USE(ip);
    VkDeviceSize at;
    if (!mesh_ring_entry(ex, &ip, &at)) return warn_once(W_MESH, "the indirect mesh ring could not be created: skipped");
    if (!prepare_draw_ex(ex, NVN_DRAW_PRIMITIVE_TRIANGLES, 0x7fffffff, 0, 1)) return;
    vk_exec_frame.draws++;
    static uint32_t n_logged;
    if (n_logged < 8 && knob_log_mesh()) {
        n_logged++;
        rt_log("vk: indirect mesh draw: arguments at %#llx, program %llu", (unsigned long long)d->ind,
               (unsigned long long)ex->prog->serial);
    }
    rec_draw_mesh_ind(fn, ex->cb, mesh_ring, at, 1, 12);
    ex->recorded++;
    vk_mem_count(ip.cls, VK_MUSE_IND, 4);
    VK_MEM_BOUND(ip);
}

static void bind_index(VkExec *ex, const void *data, uint64_t bytes, VkIndexType type) {
    VkBuffer b;
    VkDeviceSize o;
    memcpy(scratch(ex, bytes, &b, &o), data, bytes);
    vkCmdBindIndexBuffer(ex->cb, b, o, type);
    ex->recorded++;
}

static void op_draw(VkExec *ex, const p_draw *d) {
    c0_binst = (uint32_t)d->base_instance;
    if (d->count <= 0 || !prepare_draw(ex, d->prim, d->count, 0)) return;
    vk_exec_frame.draws++;
    uint32_t inst = d->instances > 0 ? (uint32_t)d->instances : 1, first = (uint32_t)d->first, n = (uint32_t)d->count;
    if (d->prim == NVN_DRAW_PRIMITIVE_QUADS && n >= 4) {   // quads become two triangles each
        uint32_t q = n / 4;
        uint32_t *ix = malloc(sizeof *ix * 6 * q);
        if (!ix) rt_fatal("vk: out of memory");
        for (uint32_t i = 0; i < q; i++) {
            uint32_t v = first + 4 * i, k[6] = {0, 1, 2, 0, 2, 3};
            for (int j = 0; j < 6; j++) ix[6 * i + j] = v + k[j];
        }
        bind_index(ex, ix, sizeof *ix * 6 * q, VK_INDEX_TYPE_UINT32);
        free(ix);
        vkCmdDrawIndexed(ex->cb, 6 * q, inst, 0, 0, (uint32_t)d->base_instance);
        mp_repeat(ex, 1, 6 * q, inst, 0, 0, 0, (uint32_t)d->base_instance);
    } else if (d->prim == NVN_DRAW_PRIMITIVE_LINE_LOOP && n >= 2) {   // a strip that returns to its first vertex
        uint32_t *ix = malloc(sizeof *ix * (n + 1));
        if (!ix) rt_fatal("vk: out of memory");
        for (uint32_t i = 0; i < n; i++) ix[i] = first + i;
        ix[n] = first;
        bind_index(ex, ix, sizeof *ix * (n + 1), VK_INDEX_TYPE_UINT32);
        free(ix);
        vkCmdDrawIndexed(ex->cb, n + 1, inst, 0, 0, (uint32_t)d->base_instance);
        mp_repeat(ex, 1, n + 1, inst, 0, 0, 0, (uint32_t)d->base_instance);
    } else {
        vkCmdDraw(ex->cb, n, inst, first, (uint32_t)d->base_instance);
        mp_repeat(ex, 0, n, inst, 0, 0, first, (uint32_t)d->base_instance);
    }
    ex->recorded++;
}

// Primitive restart of an indexed draw. NVN restarts strips wherever an index equals the SetPrimitiveRestart value;
// Vulkan restarts only at the all-ones value of the bound index type, and only strips and fans without
// primitiveTopologyListRestart. Byte indices are widened to 16 bits, so their restart value becomes 0xffff
// (op_draw_idx). Without restart the 0xffff entries of NX2NAVI's terrain and cliff strips fetch vertex 65535 + base and
// the triangles around them drop out.
static int restart_for(const VkExec *ex, int prim, int itype) {
    if (!ex->prim_restart.enable) return 0;
    switch (vk_pipeline_topology(prim)) {
    case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP: case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP: case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN:
    case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY: case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY: break;
    default: return 0;   // lists: a restart index only drops the unfinished primitive
    }
    static const uint32_t all_ones[3] = {0xffu, 0xffffu, 0xffffffffu};
    if (ex->prim_restart.index == all_ones[itype]) return 1;
    warn_once(W_RESTART, "primitive restart index %#x with %d-byte indices is not replayed: strips drawn without restart",
              ex->prim_restart.index, 1 << itype);
    return 0;
}

static void op_draw_idx(VkExec *ex, const p_draw_idx *d) {
    c0_binst = (uint32_t)d->base_instance;
    static const uint32_t isize[3] = {1, 2, 4};
    if (d->itype < 0 || d->itype > 2) return warn_once(W_INDEX, "index type %d is unknown: draw skipped", d->itype);
    int restart = restart_for(ex, d->prim, d->itype);
    if (d->count <= 0 || !prepare_draw(ex, d->prim, d->count, restart)) return;
    vk_exec_frame.draws++;
    uint64_t bytes = (uint64_t)d->count * isize[d->itype];
    VkMemSpan sp;
    if (!vk_mem_resolve(d->ibuf, bytes, &sp)) return warn_once(W_INDEX, "index buffer %#llx is not backed: draws with it are skipped", (unsigned long long)d->ibuf);
    if (d->itype == NVN_INDEX_TYPE_UNSIGNED_BYTE) {   // no uint8 index type here: widen (host-readable backing only)
        if (sp.alloc) return warn_once(W_INDEX, "byte indices in device-local memory: draw skipped");
        vk_mem_count(sp.cls, VK_MUSE_CPU_READ, bytes);
        vk_mem_count(VK_MCLS_SCRATCH, VK_MUSE_IB, 2 * bytes);
        uint16_t *ix = malloc(sizeof *ix * (size_t)d->count);
        if (!ix) rt_fatal("vk: out of memory");
        for (int i = 0; i < d->count; i++) {
            uint8_t v = ((const uint8_t *)(uintptr_t)d->ibuf)[i];
            ix[i] = restart && v == 0xff ? 0xffff : v;
        }
        bind_index(ex, ix, sizeof *ix * (size_t)d->count, VK_INDEX_TYPE_UINT16);
        free(ix);
    } else if (sp.offset % isize[d->itype]) {
        return warn_once(W_INDEX, "index buffer %#llx is not aligned to its index size: draw skipped", (unsigned long long)d->ibuf);
    } else {
        VK_MEM_USE(sp);
        vkCmdBindIndexBuffer(ex->cb, sp.buffer, sp.offset, d->itype == NVN_INDEX_TYPE_UNSIGNED_SHORT ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32);
        ex->recorded++;
        vk_mem_count(sp.cls, VK_MUSE_IB, bytes);
        VK_MEM_BOUND(sp);
    }
    vkCmdDrawIndexed(ex->cb, (uint32_t)d->count, d->instances > 0 ? (uint32_t)d->instances : 1, 0, d->base_vertex,
                     (uint32_t)d->base_instance);
    // the same draw for every remote player (multiplayer.c); byte indices were widened above and keep the draw's own
    mp_repeat(ex, 1, (uint32_t)d->count, d->instances > 0 ? (uint32_t)d->instances : 1, 0, d->base_vertex, 0,
              (uint32_t)d->base_instance);
    ex->recorded++;
    if (d->itype != NVN_INDEX_TYPE_UNSIGNED_BYTE) mvobj_draw(ex, d, sp.buffer, sp.offset);
}

// DrawElementsIndirect: the arguments are read by the GPU from the pool buffer (they are often written by an earlier
// dispatch). The base instance stays out of c0 (0x644): c0 cannot be patched by a transfer
// inside a pass; gl_BaseInstance of the Vulkan draw is the argument's. Byte indices need the count to be widened: skipped.
static void op_draw_idx_ind(VkExec *ex, const p_draw_idx_ind *d) {
    c0_binst = 0;
    if (d->itype != NVN_INDEX_TYPE_UNSIGNED_SHORT && d->itype != NVN_INDEX_TYPE_UNSIGNED_INT)
        return warn_once(W_INDEX, "indirect draw with index type %d: skipped", d->itype);
    VkMemSpan ib, ip;
    if (!vk_mem_resolve(d->ind, sizeof(VkDrawIndexedIndirectCommand), &ip) || ip.offset % 4)
        return warn_once(W_INDIRECT, "indirect draw arguments at %#llx are not backed or not aligned: draws skipped", (unsigned long long)d->ind);
    if (!vk_mem_resolve(d->ibuf, 1, &ib) || ib.offset % (d->itype == NVN_INDEX_TYPE_UNSIGNED_SHORT ? 2 : 4))
        return warn_once(W_INDEX, "index buffer %#llx is not backed or not aligned: indirect draws skipped", (unsigned long long)d->ibuf);
    if (!prepare_draw(ex, d->prim, 0x7fffffff, restart_for(ex, d->prim, d->itype))) return;
    vk_exec_frame.draws++;
    VK_MEM_USE(ip);
    VK_MEM_USE(ib);
    vkCmdBindIndexBuffer(ex->cb, ib.buffer, ib.offset, d->itype == NVN_INDEX_TYPE_UNSIGNED_SHORT ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexedIndirect(ex->cb, ip.buffer, ip.offset, 1, sizeof(VkDrawIndexedIndirectCommand));
    ex->recorded += 2;
    vk_mem_count(ip.cls, VK_MUSE_IND, sizeof(VkDrawIndexedIndirectCommand));
    VK_MEM_BOUND(ip);
    VK_MEM_BOUND(ib);
}

// ---- dispatch and uniform buffer updates

// A compute dispatch whose shader cannot be translated: Switch 2 titles run tensor-core networks (HMMA/LDSM, an
// AI upscaler: Ocarina's title movie goes 1920x1080 -> 2560x1440 through ten such passes) that the Maxwell translator
// has no model for. Instead of leaving the outputs untouched (black), the largest bound storage image receives a
// linear blit of the largest bound sampled RGBA8 texture that is a different image and no larger (the networks'
// 10-bit intermediates are only partly meaningful as pictures, so they are not a source): what an upscaler
// approximates. In-place passes (the output also sampled) are left alone. NX2NAVI_CS_FALLBACK=0 turns it off.
// sRGB-ness of the RGBA8 formats the fallback blits: a blit converts between encodings, the game's network writes the
// bits through a UNORM view, so a UNORM source and an sRGB target go through a scratch image in the source's format
// (blit, scaling only) and then a raw copy.
static int fmt_srgb(VkFormat f) { return f == VK_FORMAT_R8G8B8A8_SRGB || f == VK_FORMAT_B8G8R8A8_SRGB || f == VK_FORMAT_A8B8G8R8_SRGB_PACK32; }
static int fmt_rgba8(VkFormat f) {
    return f == VK_FORMAT_R8G8B8A8_UNORM || f == VK_FORMAT_R8G8B8A8_SRGB || f == VK_FORMAT_B8G8R8A8_UNORM ||
           f == VK_FORMAT_B8G8R8A8_SRGB || f == VK_FORMAT_A8B8G8R8_UNORM_PACK32 || f == VK_FORMAT_A8B8G8R8_SRGB_PACK32;
}

// Scratch images for those blits, one per (size, format), kept for the life of the process (a handful).
static VkImage fallback_scratch(VkExec *ex, uint32_t w, uint32_t h, VkFormat f) {
    static struct { uint32_t w, h; VkFormat f; VkImage img; VmaAllocation a; } cache[8];
    static int n;
    for (int i = 0; i < n; i++) if (cache[i].w == w && cache[i].h == h && cache[i].f == f) return cache[i].img;
    if (n == 8) return VK_NULL_HANDLE;
    VkCtx *c = vk_ctx();
    VkImageCreateInfo ii = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = f,
                            .extent = {w, h, 1}, .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
                            .tiling = VK_IMAGE_TILING_OPTIMAL,
                            .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                     (f == VK_FORMAT_E5B9G9R9_UFLOAT_PACK32 && knob_tm_dispatch() != 2 ? VK_IMAGE_USAGE_SAMPLED_BIT : 0),
                            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    VmaAllocationCreateInfo ia = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    VkImage img; VmaAllocation a;
    if (vmaCreateImage(c->vma, &ii, &ia, &img, &a, NULL) != VK_SUCCESS) return VK_NULL_HANDLE;
    VkImageMemoryBarrier2 bar = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                 .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                 .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                                 .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                 .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                 .image = img, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &bar};
    vkCmdPipelineBarrier2(ex->cb, &di);
    cache[n].w = w, cache[n].h = h, cache[n].f = f, cache[n].img = img, cache[n].a = a, n++;
    return img;
}

// 1 when the bound compute program uses tensor-core instructions (HMMA 0x23c, IMMA 0x237, LDSM 0x83b): the game's
// upscalers (the movie's and the 3D scene's), which the translator cannot run and the fallback blit stands in for.
// Other untranslated programs (bindless texturing, indirect branches) bind the output image too, and a blit for them
// pasted G-buffer normals and intermediates over the picture. Kept per program serial.
static int uses_tensor_ops(const VkExec *ex) {
    static struct { uint64_t serial; int8_t v; } memo[256];
    const uint8_t *ctl, *code;
    uint32_t len;
    if (!ex->prog || !find_stage(ex->prog, NVN_STAGE_CS, &ctl, &code, &len)) return 0;
    uint64_t sr = ex->prog->serial;
    typeof(memo[0]) *m = &memo[(sr * 0x9E3779B97F4A7C15ull) >> 56];
    if (sr && m->serial == sr) return m->v;
    int v = 0;
    if (ctl_nvn2(ctl))
        for (uint32_t o = 0; o + 16 <= len && !v; o += 16) {
            uint32_t op = (uint32_t)(code[o] | code[o + 1] << 8) & 0xfff;
            v = op == 0x23c || op == 0x237 || op == 0x83b;
        }
    m->serial = sr, m->v = (int8_t)v;
    return v;
}

// NX2NAVI_NATIVE_TENSOR=1: tensor-core programs run their translation (HMMA, LDSM and MOVM lowered to warp shuffles in
// tools/xlate/AmpereLowering.cs Hmma, Ldsm, Movm). Off by default: the lowered networks do not match the hardware yet
// (Ocarina's scene upscaler wrote infinities and noise), so they take the fallback below as untranslated ones do.
static int native_tensor(const VkExec *ex) {
    static int on = -1;
    if (on < 0) on = rt_knob("NATIVE_TENSOR", "0");
    return on || !uses_tensor_ops(ex);
}

// Everything an untranslated program binds (NX2NAVI_LOG_CS_BINDS): what a stand-in could use instead. Texture lock held.
static void log_cs_bindings_locked(const VkExec *ex) {
    char line[1600];
    int o = 0;
    line[0] = 0;
    for (int i = 0; i < 160 && o < (int)sizeof line - 80; i++) {
        int im = i >= 128;
        uint64_t h = im ? ex->img[NVN_STAGE_CS][i - 128 < 32 ? i - 128 : 0] : ex->tex[NVN_STAGE_CS][i];
        if (im && i - 128 >= 32) break;
        view_t v; int hv = 0;
        const VkTexImage *t = h ? vk_texture_by_id_locked(NVN_HANDLE_TEX(h), &v, &hv) : NULL;
        if (t && (i == 0 || h != ex->tex[NVN_STAGE_CS][0] || im))
            o += snprintf(line + o, sizeof line - o, " %s%d=%ux%u/f%d@%#llx", im ? "img" : "tex", im ? i - 128 : i,
                          t->width, t->height, t->format, (unsigned long long)t->addr);
    }
    for (int i = 0; i < 18 && o < (int)sizeof line - 60; i++)
        if (ex->ubo[NVN_STAGE_CS][i].size)
            o += snprintf(line + o, sizeof line - o, " ubo%d=%#llx+%#x", i + 3,
                          (unsigned long long)ex->ubo[NVN_STAGE_CS][i].addr, (unsigned)ex->ubo[NVN_STAGE_CS][i].size);
    rt_log("vk_exec:   its bindings:%s", line);
}

// NX2NAVI_LOG_CS_BINDS: an untranslated program the fallback leaves alone, so its output keeps what it held (black when
// nothing else wrote it): why, and what it binds. in_place -1: not a tensor program (its translation failed). Each
// (program, output) once, 256 at most. Texture lock held.
static void log_fallback_skip_locked(const VkExec *ex, const VkTexImage *out, int in_place) {
    static struct { uint64_t prog, d; } seen[256];
    static int n;
    uint64_t pr = ex->prog ? ex->prog->serial : 0, d = out ? out->addr : 0;
    for (int k = 0; k < n; k++)
        if (seen[k].prog == pr && seen[k].d == d) return;
    if (n == 256) return;
    seen[n].prog = pr, seen[n].d = d, n++;
    if (in_place < 0)
        rt_log("vk_exec: fallback leaves prog %llu alone: not translated and no tensor-core ops, so nothing stands in",
               (unsigned long long)pr);
    else if (!out)
        rt_log("vk_exec: fallback leaves prog %llu alone: it binds no single-sample 2D colour image to write",
               (unsigned long long)pr);
    else
        rt_log("vk_exec: fallback leaves prog %llu alone: %s; its output %ux%u fmt %d at %#llx", (unsigned long long)pr,
               in_place ? "it also samples its output (in place)"
                        : "no RGBA8 input and no input in the output's format, no larger than the output",
               out->width, out->height, out->format, (unsigned long long)out->addr);
    log_cs_bindings_locked(ex);
}

static void untranslated_dispatch_fallback(VkExec *ex) {
    static int on = -1;
    if (on < 0) on = rt_knob("CS_FALLBACK", "1");
    if (!on) return;
    if (!uses_tensor_ops(ex)) {   // its translation failed, and no stand-in exists for programs without tensor ops
        if (knob_log_cs_binds()) {
            vk_texture_lock();
            log_fallback_skip_locked(ex, NULL, -1);
            vk_texture_unlock();
        }
        return;
    }
    const VkTexImage *dst = NULL, *src = NULL, *out;
    int in_place = 0;
    vk_texture_lock();
    for (int i = 0; i < 32; i++) {
        uint64_t h = ex->img[NVN_STAGE_CS][i];
        view_t v; int hv = 0;
        const VkTexImage *t = h ? vk_texture_by_id_locked(NVN_HANDLE_TEX(h), &v, &hv) : NULL;
        if (t && t->aspect == VK_IMAGE_ASPECT_COLOR_BIT && t->type == VK_IMAGE_TYPE_2D && t->samples == 1 &&
            (!dst || (uint64_t)t->width * t->height > (uint64_t)dst->width * dst->height))
            dst = t;
    }
    out = dst;
    for (int i = 0; dst && i < 128; i++) {   // an in-place pass (its output also sampled) is left alone
        uint64_t h = ex->tex[NVN_STAGE_CS][i];
        view_t v; int hv = 0;
        const VkTexImage *t = h ? vk_texture_by_id_locked(NVN_HANDLE_TEX(h), &v, &hv) : NULL;
        if (t && t->image == dst->image) { dst = NULL; in_place = 1; break; }
    }
    for (int i = 0; dst && i < 128; i++) {
        uint64_t h = ex->tex[NVN_STAGE_CS][i];
        view_t v; int hv = 0;
        const VkTexImage *t = h ? vk_texture_by_id_locked(NVN_HANDLE_TEX(h), &v, &hv) : NULL;
        // RGBA8 sources (the movie's frames; its 10-bit intermediates gave black bands), or one in the output's own
        // format: the 3D scene (r11g11b10f) the game's upscaler enlarges to the output resolution
        int rgba8 = t && (t->format == VK_FORMAT_R8G8B8A8_UNORM || t->format == VK_FORMAT_R8G8B8A8_SRGB);
        int same = t && t->format == dst->format;
        if (t && (rgba8 || same) && t->image != dst->image && t->aspect == VK_IMAGE_ASPECT_COLOR_BIT && t->type == VK_IMAGE_TYPE_2D &&
            t->samples == 1 && t->width <= dst->width && t->height <= dst->height && t->width >= 16 && t->height >= 16 &&
            (!src || (uint64_t)t->width * t->height > (uint64_t)src->width * src->height))
            src = t;
    }
    if (dst && src) vk_texture_storage_bound_locked(dst);
    else if (knob_log_cs_binds()) log_fallback_skip_locked(ex, out, in_place);
    if (dst && src && knob_log_cs_binds()) {   // each distinct (program, source, target) once, 256 at most
        static struct { uint64_t prog, s, d; } seen[256];
        static int n;
        uint64_t pr = ex->prog ? ex->prog->serial : 0;
        int k = 0;
        while (k < n && (seen[k].prog != pr || seen[k].s != src->addr || seen[k].d != dst->addr)) k++;
        if (k == n && n < 256) seen[n++] = (typeof(seen[0])){pr, src->addr, dst->addr}, k = -1;
        if (k < 0) {
            rt_log("vk_exec: fallback blit prog %llu: %ux%u fmt %d at %#llx -> %ux%u fmt %d at %#llx",
                   (unsigned long long)(ex->prog ? ex->prog->serial : 0), src->width, src->height, src->format,
                   (unsigned long long)src->addr, dst->width, dst->height, dst->format, (unsigned long long)dst->addr);
            log_cs_bindings_locked(ex);
        }
    }
    VkImage si = src ? src->image : VK_NULL_HANDLE, di = dst ? dst->image : VK_NULL_HANDLE;
    VkImageView dst_view = VK_NULL_HANDLE;   // for the tonemap pass, while the lock keeps the image
    if (src && dst && (dst->usage & VK_IMAGE_USAGE_STORAGE_BIT) && raw_9e5(src->image, src->format) &&
        dst->format == VK_FORMAT_B10G11R11_UFLOAT_PACK32 && knob_tm_dispatch() != 2) {
        view_t nv;
        memset(&nv, 0, sizeof nv);
        dst_view = get_view(ex, dst, &nv, 0, 3, 1);   // 3: the translator's Texture2D
    }
    uint32_t sw = src ? src->width : 0, sh = src ? src->height : 0, dw = dst ? dst->width : 0, dh = dst ? dst->height : 0;
    VkFormat sf = src ? src->format : VK_FORMAT_UNDEFINED, df = dst ? dst->format : VK_FORMAT_UNDEFINED;
    vk_texture_unlock();
    if (!si || !di) return;
    static int logged;
    if (!logged++) rt_log("vk_exec: untranslated compute dispatch: its output image (%ux%u) gets a linear blit of its "
                          "largest input (%ux%u) instead (NX2NAVI_CS_FALLBACK=0 turns this off)", dw, dh, sw, sh);
    vk_exec_pass_end(ex);
    vk_exec_barrier(ex);
    VkImageBlit bl = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}};
    bl.srcOffsets[1] = (VkOffset3D){(int32_t)sw, (int32_t)sh, 1};
    bl.dstOffsets[1] = (VkOffset3D){(int32_t)dw, (int32_t)dh, 1};
    VkImage scratch = fmt_rgba8(df) && fmt_srgb(sf) != fmt_srgb(df) ? fallback_scratch(ex, dw, dh, sf) : VK_NULL_HANDLE;
    static int e5_blit = -1;   // E5B9G9R9 can be a blit source (with linear filtering)
    if (e5_blit < 0) {
        VkFormatProperties fp;
        vkGetPhysicalDeviceFormatProperties(vk_ctx()->phys, VK_FORMAT_E5B9G9R9_UFLOAT_PACK32, &fp);
        VkFormatFeatureFlags need = VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
        e5_blit = (fp.optimalTilingFeatures & need) == need ? 1 + !!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) : 0;
        rt_log("vk_exec: RGB9E5 fallback decode %s", e5_blit ? "on" : "unavailable (E5B9G9R9 is no blit source here)");
    }
    VkImage e5 = !scratch && e5_blit && raw_9e5(si, sf) ? fallback_scratch(ex, sw, sh, VK_FORMAT_E5B9G9R9_UFLOAT_PACK32) : VK_NULL_HANDLE;
    if (e5) {   // the raw words into a shared-exponent image (same texel size: a copy keeps the bits), then the blit
        VkImageCopy cp = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                          .extent = {sw, sh, 1}};
        vkCmdCopyImage(ex->cb, si, VK_IMAGE_LAYOUT_GENERAL, e5, VK_IMAGE_LAYOUT_GENERAL, 1, &cp);
        VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                               .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                               .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT};
        VkDependencyInfo dep = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
        vkCmdPipelineBarrier2(ex->cb, &dep);
        si = e5;
    }
#if NX2NAVI_TONEMAP_STANDIN
    // The square decode of the stored picture (vk_tonemap.c). Builds of round 24 lost the device with it: its descriptor
    // push did not go through the recorder (vk_rec.h rec_push; PORT-HANDOFF 7.12 item 11), which it now does.
    VkImage tm = e5 && dst_view ? vk_tonemap_record(ex->cb, e5, VK_FORMAT_E5B9G9R9_UFLOAT_PACK32, dw, dh) : VK_NULL_HANDLE;
    if (tm) {   // the tonemapped picture, already at the target's size, then the same blit as below
        B.pipe[1] = VK_NULL_HANDLE;   // the replay must bind its compute pipeline again
        si = tm, sw = dw, sh = dh;
        bl.srcOffsets[1] = (VkOffset3D){(int32_t)sw, (int32_t)sh, 1};
    }
#else
    VkImage tm = VK_NULL_HANDLE;
    (void)dst_view;
#endif
    // The scene upscaler's input is the post pass's picture stored as the square root of each linear value (prog 102
    // packs sqrt(composite) into RGB9E5), and its output is linear: the HUD is drawn over it and the frame's last pass
    // encodes to sRGB. A plain blit carried the stored values into the linear target, so the picture was washed out.
    // The pass above squares them. Without it, this approximation decodes with the sRGB curve instead: scale into an
    // RGBA8 UNORM scratch (the stored values), copy those bits into an RGBA8 sRGB scratch, and blit that into the
    // target, which reads the sRGB image as linear (mid-tones a little dark, the darkest values a little lifted).
    int filter_nearest = tm || (e5 && e5_blit == 1);
    VkImage g8 = VK_NULL_HANDLE, g8s = VK_NULL_HANDLE;
    if (e5 && !tm && !scratch && !fmt_rgba8(df)) {
        g8 = fallback_scratch(ex, dw, dh, VK_FORMAT_R8G8B8A8_UNORM);
        g8s = g8 ? fallback_scratch(ex, dw, dh, VK_FORMAT_R8G8B8A8_SRGB) : VK_NULL_HANDLE;
    }
    if (g8 && g8s) {
        VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                               .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                               .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT};
        VkDependencyInfo dep = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
        vkCmdBlitImage(ex->cb, si, VK_IMAGE_LAYOUT_GENERAL, g8, VK_IMAGE_LAYOUT_GENERAL, 1, &bl,
                       filter_nearest ? VK_FILTER_NEAREST : VK_FILTER_LINEAR);
        vkCmdPipelineBarrier2(ex->cb, &dep);
        VkImageCopy cp = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                          .extent = {dw, dh, 1}};
        vkCmdCopyImage(ex->cb, g8, VK_IMAGE_LAYOUT_GENERAL, g8s, VK_IMAGE_LAYOUT_GENERAL, 1, &cp);
        vkCmdPipelineBarrier2(ex->cb, &dep);
        si = g8s, sw = dw, sh = dh, filter_nearest = 1;
        bl.srcOffsets[1] = (VkOffset3D){(int32_t)sw, (int32_t)sh, 1};
    }
    vkCmdBlitImage(ex->cb, si, VK_IMAGE_LAYOUT_GENERAL, scratch ? scratch : di, VK_IMAGE_LAYOUT_GENERAL, 1, &bl,
                   filter_nearest ? VK_FILTER_NEAREST : VK_FILTER_LINEAR);
    if (scratch) {   // the bits as they are: a UNORM picture stays the same picture in an sRGB image's memory
        VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                               .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                               .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT};
        VkDependencyInfo dep = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
        vkCmdPipelineBarrier2(ex->cb, &dep);
        VkImageCopy cp = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                          .extent = {dw, dh, 1}};
        vkCmdCopyImage(ex->cb, scratch, VK_IMAGE_LAYOUT_GENERAL, di, VK_IMAGE_LAYOUT_GENERAL, 1, &cp);
    }
    ex->recorded++;
    vk_exec_barrier(ex);
}

static void op_dispatch(VkExec *ex, const p_dispatch *d) {
    if (d->x <= 0 || d->y <= 0 || d->z <= 0) return;
    if (!ex->prog) return warn_once(W_PROGRAM, "a dispatch without a bound program is skipped");
    vk_contact_flush(ex);
    cb_res(ex);
    dummies_init();
    int present;
    const GfxShader *cs = shader_for(ex, NVN_STAGE_CS, &present, NULL, 0);   // synchronous: a skipped dispatch changes results
    if (knob_log_cs_binds()) {   // diagnosis: the compute bindings of the first dispatch of each program
        static uint64_t seen[64]; static int nseen;
        int known = 0;
        for (int i = 0; i < nseen; i++) known |= seen[i] == ex->prog->serial;
        if (!known && nseen < 64) {
            seen[nseen++] = ex->prog->serial;
            char line[2048]; int o = snprintf(line, sizeof line, "cs binds prog %llu %dx%dx%d:", (unsigned long long)ex->prog->serial, d->x, d->y, d->z);
            for (int i = 0; i < 128 && o < 1900; i++) if (ex->tex[NVN_STAGE_CS][i]) o += snprintf(line + o, sizeof line - o, " tex%d=%llx", i, (unsigned long long)ex->tex[NVN_STAGE_CS][i]);
            for (int i = 0; i < 32 && o < 1900; i++) if (ex->img[NVN_STAGE_CS][i]) o += snprintf(line + o, sizeof line - o, " img%d=%llx", i, (unsigned long long)ex->img[NVN_STAGE_CS][i]);
            const GfxShaderRes *rs;
            int nt = cs ? gfx_shader_resources(cs, GFX_RES_TEX, &rs) : 0;
            for (int i = 0; i < nt && o < 1900; i++) o += snprintf(line + o, sizeof line - o, " want-tex(h%u,slot%d)", rs[i].handle, rs[i].slot);
            int ni = cs ? gfx_shader_resources(cs, GFX_RES_IMG, &rs) : 0;
            for (int i = 0; i < ni && o < 1900; i++) o += snprintf(line + o, sizeof line - o, " want-img(h%u,slot%d)", rs[i].handle, rs[i].slot);
            rt_log("%s%s", line, cs ? "" : " (not translated)");
        }
    }
    if (!cs || !native_tensor(ex)) { untranslated_dispatch_fallback(ex); return; }
    const VkPipeInfo *info = NULL;
    VkPipeline p = vk_pipeline_compute(cs, &info);
    if (!p || !info) return warn_once(W_PIPELINE, "a compute pipeline could not be created: dispatches with it are skipped");
    if ((info->pool && !pool_on()) || shader_skipped(cs)) return;   // NX2NAVI_POOL=0, NX2NAVI_SKIP_SHADERS
    vk_exec_barrier(ex);   // ends the pass; earlier writes are visible to the shader
    bind_pipeline(ex, VK_PIPELINE_BIND_POINT_COMPUTE, p);
    c0_dim[0] = (uint32_t)d->x, c0_dim[1] = (uint32_t)d->y, c0_dim[2] = (uint32_t)d->z;
    twin_dispatch = 1;
    bind_resources(ex, info);
    twin_dispatch = 0;
    VkPassMark *m = dispatch_mark(ex, cs, (uint32_t)d->x, (uint32_t)d->y, (uint32_t)d->z);
    vkCmdDispatch(ex->cb, (uint32_t)d->x, (uint32_t)d->y, (uint32_t)d->z);
    vk_pass_mark_end(ex->cb, m);
    ex->recorded++;
    if (vk_dump_armed) vk_dump_dispatch(ex, cs, p, info, d);
}

// DispatchComputeIndirect: vkCmdDispatchIndirect from the pool buffer. The shader reads the group counts from c0 0x200,
// so they are copied there on the GPU, into a c0 range no other dispatch uses, before the dispatch.
static void op_dispatch_ind(VkExec *ex, const p_dispatch_ind *d) {
    if (!ex->prog) return warn_once(W_PROGRAM, "a dispatch without a bound program is skipped");
    VkMemSpan ip;
    if (!vk_mem_resolve(d->ind, sizeof(VkDispatchIndirectCommand), &ip) || ip.offset % 4)
        return warn_once(W_INDIRECT, "indirect dispatch arguments at %#llx are not backed or not aligned: skipped", (unsigned long long)d->ind);
    cb_res(ex);
    dummies_init();
    int present;
    const GfxShader *cs = shader_for(ex, NVN_STAGE_CS, &present, NULL, 0);
    if (!cs || !native_tensor(ex)) { untranslated_dispatch_fallback(ex); return; }
    const VkPipeInfo *info = NULL;
    VkPipeline p = vk_pipeline_compute(cs, &info);
    if (!p || !info) return warn_once(W_PIPELINE, "a compute pipeline could not be created: dispatches with it are skipped");
    if ((info->pool && !pool_on()) || shader_skipped(cs)) return;   // NX2NAVI_POOL=0, NX2NAVI_SKIP_SHADERS
    vk_exec_barrier(ex);
    VK_MEM_USE(ip);
    bind_pipeline(ex, VK_PIPELINE_BIND_POINT_COMPUTE, p);
    c0_dim[0] = c0_dim[1] = c0_dim[2] = 0;
    B.c0_valid[2] = 0;   // a fresh c0 range: the copy below must not change one an earlier dispatch reads
    twin_dispatch = 1;
    bind_resources(ex, info);
    twin_dispatch = 0;
    if (B.c0_valid[2]) {   // the shader binds c0
        // NVN2 programs read the group counts from c0 byte 0 (c0_info); at 0x200 their bank holds pushed constants
        // or the driver's tables, which the copy would overwrite
        const uint8_t *pctl, *pcode;
        uint32_t plen;
        int nvn2 = find_stage(ex->prog, NVN_STAGE_CS, &pctl, &pcode, &plen) && ctl_nvn2(pctl);
        VkBufferCopy cp = {ip.offset, B.c0_off[2] + (nvn2 ? 0 : 0x200), sizeof(VkDispatchIndirectCommand)};
        vkCmdCopyBuffer(ex->cb, ip.buffer, B.c0_buf[2], 1, &cp);
        ex->recorded++;
        vk_exec_barrier(ex);
        B.c0_valid[2] = 0;   // the range no longer holds B.c0[2]
    }
    VkPassMark *m = dispatch_mark(ex, cs, 0, 0, 0);
    vkCmdDispatchIndirect(ex->cb, ip.buffer, ip.offset);
    vk_pass_mark_end(ex->cb, m);
    ex->recorded++;
    vk_mem_count(ip.cls, VK_MUSE_IND, sizeof(VkDispatchIndirectCommand));
    VK_MEM_BOUND(ip);
    if (vk_dump_armed) vk_dump_dispatch(ex, cs, p, info, &(p_dispatch){0, 0, 0, d->caller});
}

static void op_update_ub(VkExec *ex, const p_update_ub *u) {
    if (!u->size) return;
    VkMemSpan sp;
    if (!vk_mem_resolve(u->dst, u->size, &sp)) return warn_once(W_UB, "UpdateUniformBuffer to %#llx is not backed: dropped", (unsigned long long)u->dst);
    cb_res(ex);
    vk_exec_frame.ub_updates++;
    if (ex->in_pass) vk_exec_frame.ub_pass_breaks++;
    vk_exec_barrier(ex);   // earlier draws finished with the old contents
    VkBuffer b;
    VkDeviceSize o;
    memcpy(scratch(ex, u->size, &b, &o), u->data, u->size);
    VK_MEM_PICK(sp);
    VkBufferCopy cp = {o, sp.offset, u->size};
    vkCmdCopyBuffer(ex->cb, b, sp.buffer, 1, &cp);
    vk_mem_gpu_wrote(&sp, u->size);
    vk_usnap_drop(u->dst, u->size);
    ex->recorded++;
    vk_mem_count(sp.cls, VK_MUSE_GPU_WRITE, u->size);
    note_written(ex, u->dst, u->size);
    upd_note(ex, u);
}

// ---- bind ops

static void bind_handles(uint64_t *dst, int cap, const p_bind_handles *b) {
    if (b->stage < 0 || b->stage >= 8 || b->first < 0 || b->n < 0 || b->n > 32 || b->first + b->n > cap) {
        warn_once(W_BIND, "a bind of %d handles at %d of stage %d is out of range: ignored", b->n, b->first, b->stage);
        return;
    }
    memcpy(&dst[b->first], b->h, sizeof(uint64_t) * (size_t)b->n);
}

void vk_exec_op_draw(VkExec *ex, uint32_t op, const void *payload) {
    switch (op) {
    case OP_BIND_PROGRAM: {
        const p_bind_program *b = payload;
        ex->prog = b->prog, ex->prog_stages = b->stages;
        bind_gen++;
        break;
    }
    case OP_BIND_STATE: {
        const p_bind_state *b = payload;
        blend_st bs;
        switch (b->kind) {
        case 0:
            memcpy(&bs, b->bytes, sizeof bs);
            if (bs.target < 8) ex->pipe.blend[bs.target] = bs;
            break;
        case 1: memcpy(&ex->pipe.color, b->bytes, sizeof ex->pipe.color); break;
        case 2: memcpy(&ex->pipe.chmask, b->bytes, sizeof ex->pipe.chmask); break;
        case 3: memcpy(&ex->pipe.ms, b->bytes, sizeof ex->pipe.ms); break;
        case 4: memcpy(&ex->pipe.poly, b->bytes, sizeof ex->pipe.poly); break;
        case 5: memcpy(&ex->pipe.ds, b->bytes, sizeof ex->pipe.ds); break;
        case 6: memcpy(&ex->point_size, b->bytes, sizeof ex->point_size); break;
        }
        break;
    }
    case OP_BIND_VATTR: {
        const p_bind_vattr *b = payload;
        if (b->first < 0 || b->n < 0 || b->first + b->n > 16) break;
        memset(ex->pipe.vattr, 0, sizeof ex->pipe.vattr);   // the new list replaces the old one
        memcpy(&ex->pipe.vattr[b->first], b->st, sizeof(uint32_t) * (size_t)b->n);
        break;
    }
    case OP_BIND_VSTREAM: {
        const p_bind_vstream *b = payload;
        if (b->first < 0 || b->n < 0 || b->first + b->n > 16) break;
        memset(ex->pipe.vstream, 0, sizeof ex->pipe.vstream);
        memcpy(&ex->pipe.vstream[b->first], b->st, sizeof(vs_st) * (size_t)b->n);
        break;
    }
    case OP_BIND_VBUF: {
        const p_bind_vbuf *b = payload;
        if (b->first < 0 || b->n < 0 || b->first + b->n > 16) break;
        memcpy(&ex->vbuf[b->first], b->r, sizeof(NVNbufferRange) * (size_t)b->n);
        break;
    }
    case OP_BIND_UBO: {
        const p_bind_buf *b = payload;
        if (b->stage >= 0 && b->stage < 8 && b->index >= 0 && b->index < 18) ex->ubo[b->stage][b->index] = *b;
        else if (b->index >= 18 && b->index < 64) {   // NVN2 has more banks than Maxwell's 18: say which, once each
            static uint64_t seen;
            if (!(__atomic_fetch_or(&seen, 1ull << b->index, __ATOMIC_RELAXED) >> b->index & 1))
                rt_log("vk: uniform buffer bind to index %d (bank %d), stage %d, %llu bytes: beyond Maxwell's banks, dropped",
                       b->index, b->index + 3, b->stage, (unsigned long long)b->size);
        }
        bind_gen++;
        break;
    }
    case OP_BIND_SSBO: {
        const p_bind_buf *b = payload;
        if (b->stage >= 0 && b->stage < 8 && b->index >= 0 && b->index < 16) ex->ssbo[b->stage][b->index] = *b;
        if (b->stage >= NVN_STAGE_TASK && b->stage <= NVN_STAGE_MESH && b->index >= 0 && b->index < 16) {   // once per slot
            static uint32_t seen;
            uint32_t bit = 1u << (16 * (b->stage - NVN_STAGE_TASK) + b->index);
            if (!(__atomic_fetch_or(&seen, bit, __ATOMIC_RELAXED) & bit))
                rt_log("vk: storage buffer bind, stage %d slot %d: %#llx, %llu bytes", b->stage, b->index,
                       (unsigned long long)b->addr, (unsigned long long)b->size);
        }
        bind_gen++;
        break;
    }
    case OP_BIND_TEX: {
        const p_bind_handles *b = payload;
        bind_handles(b->stage >= 0 && b->stage < 8 ? ex->tex[b->stage] : NULL, 128, b);
        ex->tex_serial++;
        bind_gen++;
        break;
    }
    case OP_BIND_SAMP: {
        const p_bind_handles *b = payload;
        bind_handles(b->stage >= 0 && b->stage < 8 ? ex->samp[b->stage] : NULL, 128, b);
        break;
    }
    case OP_BIND_IMG: {
        const p_bind_handles *b = payload;
        bind_handles(b->stage >= 0 && b->stage < 8 ? ex->img[b->stage] : NULL, 32, b);
        ex->tex_serial++;   // the shader memo's handle check (feedback_scan reads only textures: a rescan at most)
        bind_gen++;
        break;
    }
    case OP_UPDATE_UB: op_update_ub(ex, payload); break;
    case OP_PUSH_CC: {
        const p_push_cc *p = payload;
        if (p->offset + (uint64_t)p->size <= sizeof ex->cs_push) memcpy(ex->cs_push + p->offset, p->data, p->size);
        cs_push_bytes = p->offset + p->size;
        bind_gen++;   // storage buffers may come from these words (push_ptr)
        break;
    }
    case OP_DRAW: op_draw(ex, payload); break;
    case OP_DRAW_MESH: vk_exec_frame.draws_seen++; op_draw_mesh(ex, payload); break;
    case OP_DRAW_MESH_IND: vk_exec_frame.draws_seen++; op_draw_mesh_ind(ex, payload); break;
    case OP_DRAW_IDX: op_draw_idx(ex, payload); break;
    case OP_DISPATCH: op_dispatch(ex, payload); break;
    case OP_DRAW_IDX_IND: op_draw_idx_ind(ex, payload); break;
    case OP_DISPATCH_IND: op_dispatch_ind(ex, payload); break;
    default: rt_fatal("vk: op %s is not a draw op", nvn_op_name(op));
    }
}

// The last executed list's draws as " <position banks>/<changed banks>:<draws>" (bit n: cn), for NX2NAVI_MV_CHECK's lines.
const char *vk_mvobj_frame_masks(void) { return fm_last; }
