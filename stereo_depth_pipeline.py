#!/usr/bin/env python3
"""
stereo_depth.py - clean depth map from a calibrated stereo pair (e.g. 2x OV7670, 640x480).

Pipeline
  1. Load calibration (mtx_l, dist_l, mtx_r, dist_r, R, T) and rectify both images
  2. Diagnostics: check row alignment / disparity sign with ORB matches (finds calibration problems)
  3. Preprocess: denoise -> histogram-match right to left -> mild CLAHE -> light blur
  4. StereoSGBM (3-way) with numDisparities chosen from your data / geometry
  5. Confidence filtering: WLS filter (opencv-contrib) or left-right consistency check (fallback)
  6. Speckle removal, hole filling, horizontal-median (kills vertical stripes), 2D median, bilateral
  7. Disparity -> depth (Z = f*B/d), save results, plot rectified pair + 2D depth + 3D depth

Usage
  python stereo_depth.py left.jpg right.jpg
  python stereo_depth.py left.jpg right.jpg --min-depth 300 --block-size 9
  python stereo_depth.py --help        (every parameter below is a command-line option)

Depth units = units of T in your calibration file (mm if you gave the square size in mm).

Install:  pip install numpy matplotlib opencv-contrib-python
          (opencv-contrib is needed for the WLS filter; the script falls back without it)
"""
import argparse
import os
import time
from dataclasses import dataclass, fields

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)

# ----------------------------------------------------------------------------
# Defaults - edit here, or override on the command line
# ----------------------------------------------------------------------------
LEFT_IMAGE = "dataset/218_left.jpg"
RIGHT_IMAGE = "dataset/218_right.jpg"
CALIB_FILE = "stereo_calibration_params.npz"

HAS_XIMGPROC = hasattr(cv2, "ximgproc")
INVALID16 = -32768


@dataclass
class Config:
    # --- geometry -----------------------------------------------------------
    t_scale: float = 110.0          # multiply T by this (e.g. 10 if T is in cm and you want mm)
    alpha: float = 0.5            # rectify crop: 0 = crop to valid pixels, 1 = keep everything
    # --- preprocessing ------------------------------------------------------
    denoise: bool = False          # non-local-means denoise (OV7670 + JPEG noise)
    denoise_h: float = 6.0        # denoise strength (3-10)
    hist_match: bool = False       # match right histogram to left (brightness/gain differences)
    clahe: bool = False            # mild local contrast (strong CLAHE amplifies noise on flat areas!)
    clahe_clip: float = 1.5
    clahe_tile: int = 8
    blur_sigma: float = 0.0       # final light Gaussian blur, 0 = off
    # --- StereoSGBM ---------------------------------------------------------
    min_disp: int = -1            # -1 = auto (search band start, from ORB matches / --max-depth)
    num_disp: int = 208             # 0 = auto (band width, from ORB matches / --min-depth)
    min_depth: float = 0.0        # closest object you care about (units of T); sets the band end
    block_size: int = 7           # odd, 5-9 for 640x480 noisy sensor
    p1_mult: int = 2              # P1 = p1_mult * block^2
    p2_mult: int = 8             # P2 = p2_mult * block^2  (larger = smoother planes)
    uniqueness: int = 2
    pre_filter_cap: int = 0
    full_dp: bool = True         # True = SGBM_HH (slower, a bit better) instead of 3WAY
    # --- confidence filtering ----------------------------------------------
    wls: bool = True              # WLS filter if opencv-contrib is installed
    wls_lambda: float = 8000.0
    wls_sigma: float = 1.25
    conf_thresh: float = 0.0     # WLS confidence 0-255; raise for cleaner/sparser output
    lr_thresh: float = 1.0        # fallback left-right consistency threshold (px)
    # --- post-processing ----------------------------------------------------
    speckle_size: int = 10       # remove disparity blobs smaller than this (px), 0 = off
    speckle_diff: float = 0.3     # max disparity jump inside a blob (px)
    fill_holes: bool = False       # output: fill invalid pixels from the farther neighbour on the row
    close_ksize: int = 2          # when fill_holes is off: close pinholes up to this size, 0/1 = off
    stripe_ksize: int = 2         # horizontal median width (kills vertical stripes), 1 = off
    median_ksize: int = 2         # 2D median (3 or 5)
    bilateral: bool = False        # edge-preserving final smoothing
    max_depth: float = 0.0        # farthest depth you care about; also starts the search band here, 0 = off
    # --- calibration image size (set via --calib-size W H) ------------------
    calib_size: tuple = (480, 640)


# ----------------------------------------------------------------------------
# Calibration & rectification
# ----------------------------------------------------------------------------
def load_calibration(path, t_scale):
    data = np.load(path)
    calib = {k: np.asarray(data[k], dtype=np.float64)
             for k in ("mtx_l", "dist_l", "mtx_r", "dist_r", "R", "T")}
    calib["T"] = calib["T"].reshape(3, 1) * t_scale
    return calib


def _scale_K(K, sx, sy):
    K = K.copy()
    K[0, 0] *= sx
    K[0, 2] *= sx
    K[1, 1] *= sy
    K[1, 2] *= sy
    return K


def build_rectifier(calib, img_size, cfg):
    w, h = img_size
    sx, sy = w / cfg.calib_size[0], h / cfg.calib_size[1]
    K_l = _scale_K(calib["mtx_l"], sx, sy)
    K_r = _scale_K(calib["mtx_r"], sx, sy)
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K_l, calib["dist_l"], K_r, calib["dist_r"], (w, h), calib["R"], calib["T"],
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=cfg.alpha)
    map_l = cv2.initUndistortRectifyMap(K_l, calib["dist_l"], R1, P1, (w, h), cv2.CV_32FC1)
    map_r = cv2.initUndistortRectifyMap(K_r, calib["dist_r"], R2, P2, (w, h), cv2.CV_32FC1)

    ones = np.full((h, w), 255, np.uint8)
    roi = (cv2.remap(ones, *map_l, cv2.INTER_NEAREST) > 0) & \
          (cv2.remap(ones, *map_r, cv2.INTER_NEAREST) > 0)
    roi = cv2.erode(roi.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)

    return dict(map_l=map_l, map_r=map_r, Q=Q, f=float(P1[0, 0]),
                B=float(np.linalg.norm(calib["T"])), roi=roi)


def rectify(img, maps):
    return cv2.remap(img, maps[0], maps[1], cv2.INTER_LINEAR)


# ----------------------------------------------------------------------------
# Diagnostics: is the rectified pair really row-aligned?
# ----------------------------------------------------------------------------
def check_rectification(gl, gr):
    orb = cv2.ORB_create(4000)
    k1, d1 = orb.detectAndCompute(gl, None)
    k2, d2 = orb.detectAndCompute(gr, None)
    if d1 is None or d2 is None or len(k1) < 20 or len(k2) < 20:
        return None
    knn = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2)
    good = [p[0] for p in knn if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < 15:
        return None
    pl = np.float32([k1[m.queryIdx].pt for m in good])
    pr = np.float32([k2[m.trainIdx].pt for m in good])
    dy = pl[:, 1] - pr[:, 1]
    dx = pl[:, 0] - pr[:, 0]
    near = np.abs(dy) <= 8                      # discard gross mismatches
    if near.sum() < 15:
        return None
    return dict(n=int(near.sum()),
                median_dy=float(np.median(dy[near])),
                frac_row_ok=float(np.mean(np.abs(dy[near]) <= 1.0)),
                dx=dx[np.abs(dy) <= 2])


def choose_disp_range(cfg, f, B, diag):
    """Pick (min_disp, num_disp, source). Searching only the band that exists removes most
    false matches; ORB matches are sparse, so generous margins are added."""
    dx = diag["dx"] if diag is not None and len(diag["dx"]) >= 15 else None
    src = []
    if cfg.min_disp >= 0:
        lo = cfg.min_disp
        src.append("min user")
    elif cfg.max_depth > 0:
        lo = f * B / cfg.max_depth
        src.append("min from --max-depth")
    elif dx is not None:
        lo = max(np.percentile(dx, 1) * 0.7 - 8, 0)
        src.append("min from ORB")
    else:
        lo = 0
        src.append("min default")
    lo = int(max(np.floor(lo / 16.0) * 16, 0))

    if cfg.num_disp > 0:
        nd = cfg.num_disp
        src.append("num user")
    else:
        if cfg.min_depth > 0:
            hi = f * B / cfg.min_depth + 8
            src.append("num from --min-depth")
        elif dx is not None:
            hi = np.percentile(dx, 99) * 1.1 + 8
            src.append("num from ORB")
        else:
            hi = lo + 128
            src.append("num default")
        nd = hi - lo
    nd = int(np.clip(np.ceil(nd / 16.0) * 16, 32, 256))
    return lo, nd, ", ".join(src)


# ----------------------------------------------------------------------------
# Preprocessing
# ----------------------------------------------------------------------------
def match_histogram(src, ref):
    s_vals, s_inv, s_cnt = np.unique(src.ravel(), return_inverse=True, return_counts=True)
    r_vals, r_cnt = np.unique(ref.ravel(), return_counts=True)
    s_q = np.cumsum(s_cnt) / src.size
    r_q = np.cumsum(r_cnt) / ref.size
    mapped = np.interp(s_q, r_q, r_vals)
    return mapped[s_inv].reshape(src.shape).astype(np.uint8)


def preprocess(gl, gr, cfg):
    if cfg.denoise:
        gl = cv2.fastNlMeansDenoising(gl, None, cfg.denoise_h, 7, 21)
        gr = cv2.fastNlMeansDenoising(gr, None, cfg.denoise_h, 7, 21)
    if cfg.hist_match:
        gr = match_histogram(gr, gl)
    if cfg.clahe:
        c = cv2.createCLAHE(clipLimit=cfg.clahe_clip, tileGridSize=(cfg.clahe_tile, cfg.clahe_tile))
        gl, gr = c.apply(gl), c.apply(gr)
    if cfg.blur_sigma > 0:
        gl = cv2.GaussianBlur(gl, (0, 0), cfg.blur_sigma)
        gr = cv2.GaussianBlur(gr, (0, 0), cfg.blur_sigma)
    return gl, gr


# ----------------------------------------------------------------------------
# Stereo matching
# ----------------------------------------------------------------------------
def make_sgbm(cfg, num_disp):
    bs = cfg.block_size | 1                      # force odd
    P1 = cfg.p1_mult * bs * bs
    P2 = max(cfg.p2_mult * bs * bs, P1 + 1)
    mode = cv2.STEREO_SGBM_MODE_HH if cfg.full_dp else cv2.STEREO_SGBM_MODE_SGBM_3WAY
    return cv2.StereoSGBM_create(
        minDisparity=cfg.min_disp, numDisparities=num_disp, blockSize=bs,
        P1=P1, P2=P2, disp12MaxDiff=1, preFilterCap=cfg.pre_filter_cap,
        uniquenessRatio=cfg.uniqueness, speckleWindowSize=0, speckleRange=0, mode=mode)


def compute_disparity(L, R, cfg, num_disp):
    """Returns (raw_vis, disp, valid): raw SGBM disparity (NaN = invalid), filtered
    float disparity in pixels, and a boolean confidence mask."""
    sgbm = make_sgbm(cfg, num_disp)
    dL = sgbm.compute(L, R)                                  # int16, disparity * 16
    raw = dL.astype(np.float32) / 16.0
    raw[dL < cfg.min_disp * 16] = np.nan
    h, w = L.shape

    if HAS_XIMGPROC and cfg.wls:
        rmatcher = cv2.ximgproc.createRightMatcher(sgbm)
        dR = rmatcher.compute(R, L)
        wls = cv2.ximgproc.createDisparityWLSFilter(matcher_left=sgbm)
        wls.setLambda(cfg.wls_lambda)
        wls.setSigmaColor(cfg.wls_sigma)
        guide = cv2.GaussianBlur(L, (0, 0), 1.5)       # smooth guide: don't follow sensor noise
        dF = wls.filter(dL, guide, disparity_map_right=dR)
        conf = wls.getConfidenceMap()
        disp = dF.astype(np.float32) / 16.0
        valid = (conf >= cfg.conf_thresh) & (dF >= cfg.min_disp * 16)
    else:
        # right-view disparity via the flip trick, then left-right consistency check
        dR = sgbm.compute(np.ascontiguousarray(R[:, ::-1]), np.ascontiguousarray(L[:, ::-1]))
        dR = dR[:, ::-1].astype(np.float32) / 16.0
        disp = dL.astype(np.float32) / 16.0
        xs = np.arange(w)[None, :]
        xr = np.rint(xs - disp).astype(int)
        inside = (xr >= 0) & (xr < w)
        dR_at = np.take_along_axis(dR, np.clip(xr, 0, w - 1), axis=1)
        valid = (dL >= cfg.min_disp * 16) & inside & (np.abs(disp - dR_at) <= cfg.lr_thresh)

    return raw, disp, valid


def remove_speckles(disp, valid, size, max_diff):
    if size <= 0:
        return valid
    img = np.where(valid, np.rint(disp * 16.0), INVALID16).astype(np.int16)
    img, _ = cv2.filterSpeckles(np.ascontiguousarray(img), INVALID16, int(size), int(max_diff * 16))
    return img != INVALID16


def fill_holes(disp, valid):
    """Fill invalid pixels with the smaller (= farther) of the nearest valid values left/right."""
    h, w = disp.shape
    cols = np.tile(np.arange(w), (h, 1))
    d = np.where(valid, disp, np.nan).astype(np.float32)
    li = np.maximum.accumulate(np.where(valid, cols, 0), axis=1)
    ri = np.minimum.accumulate(np.where(valid, cols, w - 1)[:, ::-1], axis=1)[:, ::-1]
    left = np.take_along_axis(d, li, axis=1)
    right = np.take_along_axis(d, ri, axis=1)
    out = np.fmin(left, right)
    out[valid] = disp[valid]
    return np.nan_to_num(out, nan=0.0).astype(np.float32)


def hmedian(img, k):
    """1-D median along x: removes vertical stripes up to ~k/2 px wide, keeps depth edges."""
    if k <= 1:
        return img
    p = k // 2
    pad = np.pad(img, ((0, 0), (p, p)), mode="edge")
    stack = np.stack([pad[:, i:i + img.shape[1]] for i in range(k)], axis=0)
    return np.median(stack, axis=0).astype(np.float32)


def refine(disp, valid, cfg):
    # Always filter on a filled copy: zeros from invalid pixels would otherwise win the median
    # wherever fewer than ~half of the neighbours are valid and erase the whole map.
    x = fill_holes(disp, valid)
    x = hmedian(x, cfg.stripe_ksize | 1)
    if cfg.median_ksize in (3, 5):
        x = cv2.medianBlur(x, cfg.median_ksize)
    if cfg.bilateral:
        x = cv2.bilateralFilter(x, 9, 1.0, 5.0)
    return x


# ----------------------------------------------------------------------------
# Output helpers
# ----------------------------------------------------------------------------
def save_ply(path, Q, disp, color_bgr, mask):
    pts = cv2.reprojectImageTo3D(disp.astype(np.float32), Q, handleMissingValues=False)
    m = mask & np.isfinite(pts).all(axis=2)
    xyz, rgb = pts[m], color_bgr[m][:, ::-1]
    with open(path, "w") as fh:
        fh.write("ply\nformat ascii 1.0\nelement vertex %d\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                 "end_header\n" % len(xyz))
        np.savetxt(fh, np.hstack([xyz, rgb]), fmt="%.3f %.3f %.3f %d %d %d")


def show_plots(l_rect, r_rect, raw, depth, step, out_dir, clip_pct=5.0):
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad("black")
    finite = np.isfinite(depth)
    vmin, vmax = np.percentile(depth[finite], [clip_pct, 100 - clip_pct]) if finite.any() else (0, 1)

    # 1) rectified pair with horizontal guide lines: features must sit on the same line
    fig1, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, img, title in zip(axes, (l_rect, r_rect), ("Left (rectified)", "Right (rectified)")):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        for y in range(0, img.shape[0], 40):
            ax.axhline(y, color="lime", lw=0.6, alpha=0.5)
        ax.set_title(title)
        ax.axis("off")
    fig1.tight_layout()

    # 2) raw SGBM disparity vs final depth
    fig2, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    rfin = np.isfinite(raw)
    rmin, rmax = np.percentile(raw[rfin], [clip_pct, 100 - clip_pct]) if rfin.any() else (0, 1)
    im0 = axes[0].imshow(np.ma.masked_invalid(raw), cmap=cmap, vmin=rmin, vmax=rmax)
    axes[0].set_title("Raw SGBM disparity (px)")
    fig2.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(np.ma.masked_invalid(depth), cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title("Clean depth map (units of T)")
    fig2.colorbar(im1, ax=axes[1], fraction=0.046)
    for ax in axes:
        ax.axis("off")
    fig2.tight_layout()

    # 3) 3D surface
    fig3 = plt.figure(figsize=(10, 7.5))
    ax3 = fig3.add_subplot(111, projection="3d")
    d = depth[::step, ::step]
    ys, xs = np.mgrid[0:depth.shape[0]:step, 0:depth.shape[1]:step]
    Z = np.where((d >= vmin) & (d <= vmax), d, np.nan)
    ax3.plot_surface(xs, ys, Z, cmap=cmap, vmin=vmin, vmax=vmax, linewidth=0, antialiased=False)
    ax3.set_xlabel("x (px)")
    ax3.set_ylabel("y (px)")
    ax3.set_zlabel("depth")
    ax3.set_zlim(vmin, vmax)
    ax3.invert_yaxis()
    ax3.invert_zaxis()                      # near objects point up/toward you
    ax3.set_box_aspect((4, 3, 2))
    ax3.view_init(elev=35, azim=-65)
    ax3.set_title("3D depth map")
    fig3.tight_layout()

    for i, fig in enumerate((fig1, fig2, fig3), 1):
        fig.savefig(os.path.join(out_dir, "plot%d.png" % i), dpi=120)


# ----------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Clean depth map from a calibrated stereo pair.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("left", nargs="?", default=LEFT_IMAGE, help="left image")
    ap.add_argument("right", nargs="?", default=RIGHT_IMAGE, help="right image")
    ap.add_argument("--calib", default=CALIB_FILE, help="calibration .npz")
    ap.add_argument("--calib-size", type=int, nargs=2, metavar=("W", "H"), default=None,
                    help="image size used during calibration (default 640 480); intrinsics are "
                         "rescaled if your images differ")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--save-ply", action="store_true", help="also save a coloured point cloud")
    ap.add_argument("--no-plot", action="store_true", help="save plots but don't open windows")
    ap.add_argument("--clip-pct", type=float, default=5.0,
                    help="colour range of the plots = this..100-this percentile of depth")
    ap.add_argument("--surface-step", type=int, default=4, help="pixel stride for the 3D surface")
    defaults = Config()
    for f in fields(Config):
        if f.name == "calib_size":
            continue
        v = getattr(defaults, f.name)
        flag = "--" + f.name.replace("_", "-")
        if isinstance(v, bool):
            ap.add_argument(flag, action=argparse.BooleanOptionalAction, default=v)
        else:
            ap.add_argument(flag, type=type(v), default=v)
    args = ap.parse_args()
    cfg = Config(**{f.name: getattr(args, f.name) for f in fields(Config) if f.name != "calib_size"})
    if args.calib_size:
        cfg.calib_size = tuple(args.calib_size)
    return args, cfg


def main():
    args, cfg = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    cfg.block_size |= 1                          # SGBM needs an odd window

    l_bgr = cv2.imread(args.left, cv2.IMREAD_COLOR)
    r_bgr = cv2.imread(args.right, cv2.IMREAD_COLOR)
    l_bgr = cv2.rotate(l_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    r_bgr = cv2.rotate(r_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if l_bgr is None or r_bgr is None:
        raise SystemExit("Could not read images: %s / %s" % (args.left, args.right))
    if l_bgr.shape != r_bgr.shape:
        raise SystemExit("Left and right images must have the same size.")
    h, w = l_bgr.shape[:2]

    calib = load_calibration(args.calib, cfg.t_scale)
    sx, sy = w / cfg.calib_size[0], h / cfg.calib_size[1]
    if abs(sx - 1) > 1e-3 or abs(sy - 1) > 1e-3:
        print('NOTE: images are %dx%d but --calib-size is %dx%d (W H): intrinsics scaled by x%.3f / y%.3f.%s'
              % (w, h, cfg.calib_size[0], cfg.calib_size[1], sx, sy,
                 '  Different x/y factors distort the geometry - check width/height are not swapped.'
                 if abs(sx - sy) > 0.01 else ''))
    rect = build_rectifier(calib, (w, h), cfg)
    f, B = rect["f"], rect["B"]
    l_rect = rectify(l_bgr, rect["map_l"])
    r_rect = rectify(r_bgr, rect["map_r"])
    gl = cv2.cvtColor(l_rect, cv2.COLOR_BGR2GRAY)
    gr = cv2.cvtColor(r_rect, cv2.COLOR_BGR2GRAY)

    # ---- diagnostics -------------------------------------------------------
    diag = check_rectification(gl, gr)
    print("\n=== Calibration / rectification check ===")
    print("Rectified focal length f = %.1f px, baseline B = %.3f (units of T)" % (f, B))
    if diag is None:
        print("Too few ORB matches for a diagnostic (scene may be too plain).")
    else:
        print("ORB matches used: %d | median vertical error: %+.2f px | within 1 px: %.0f%%"
              % (diag["n"], diag["median_dy"], 100 * diag["frac_row_ok"]))
        if diag["frac_row_ok"] < 0.8 or abs(diag["median_dy"]) > 1:
            print("  WARNING: rows are not well aligned. Vertical stripes/noise will persist until the "
                  "calibration is improved (more checkerboard poses, covering the corners, RMS < 0.5 px).")
        if np.median(diag["dx"]) < 0:
            print("  WARNING: negative disparities - left/right images (or calibration order) are swapped.")
        else:
            print("  Matched disparity range: %.0f .. %.0f px"
                  % (np.percentile(diag["dx"], 1), np.percentile(diag["dx"], 99)))

    # ---- parameters --------------------------------------------------------
    cfg.min_disp, num_disp, src = choose_disp_range(cfg, f, B, diag)
    d_hi = cfg.min_disp + num_disp
    print("\n=== Matcher parameters ===")
    print("minDisparity = %d | numDisparities = %d (%s) | blockSize = %d | P1 = %d | P2 = %d"
          % (cfg.min_disp, num_disp, src, cfg.block_size | 1,
             cfg.p1_mult * cfg.block_size ** 2, cfg.p2_mult * cfg.block_size ** 2))
    zmin = f * B / d_hi
    zmax = f * B / cfg.min_disp if cfg.min_disp > 0 else float("inf")
    print("Searching disparities %d..%d px  ->  depth range %.4g .. %.4g (units of T)"
          % (cfg.min_disp, d_hi - 1, zmin, zmax))
    print("Depth error for a 0.5 px disparity error: %.3g at %.4g, %.3g at %.4g"
          % ((zmin * 1.5) ** 2 / (f * B) * 0.5, zmin * 1.5, (zmin * 3) ** 2 / (f * B) * 0.5, zmin * 3))
    if d_hi > 0.3 * w:
        print("  NOTE: the left %d columns (%.0f%%) cannot be matched - the scene is close for this "
              "baseline. Move farther away (disparity falls as 1/depth) or reduce the baseline."
              % (d_hi, 100.0 * d_hi / w))
    print("Filter: %s" % ("WLS (opencv-contrib)" if HAS_XIMGPROC and cfg.wls
                          else "left-right consistency check (install opencv-contrib-python for WLS)"))

    # ---- matching ----------------------------------------------------------
    t0 = time.time()
    pl, pr = preprocess(gl, gr, cfg)
    raw, disp, valid = compute_disparity(pl, pr, cfg, num_disp)
    valid = remove_speckles(disp, valid, cfg.speckle_size, cfg.speckle_diff)
    print('Confident pixels after matching/filtering: %.1f%% of the image' % (100 * valid.mean()))
    final = refine(disp, valid, cfg)

        # 1. Base geometric mask from ROI and your configuration settings
    keep = rect["roi"] & (final > 0)
    
    if cfg.fill_holes:
        margin = np.zeros_like(valid)
        margin[:, :cfg.min_disp + num_disp] = True    # left columns SGBM cannot match
        keep &= ~(margin & ~valid)
    else:
        k = cfg.close_ksize
        vm = cv2.morphologyEx(valid.astype(np.uint8), cv2.MORPH_CLOSE,
                              np.ones((k, k), np.uint8)).astype(bool) if k > 1 else valid
        keep &= vm

    # 2. Initialize depth map strictly as NaNs
    depth = np.full(final.shape, np.nan, np.float32)
    
    # 3. Safely extract valid disparities inside the mask
    # We add an extra safety check (final > 0.5) to protect against tiny numbers near zero
    valid_disparity_mask = keep & (final > 0.5)
    
    # 4. Perform the depth calculation
    depth[valid_disparity_mask] = f * B / final[valid_disparity_mask]
    
    # 5. Dynamic Safety Scale Protection (Since cfg.max_depth is 0)
    # If max_depth is 0, we automatically filter out values higher than the 98th percentile
    # This completely stops the "giant blue plain" scale explosion bug.
    if cfg.max_depth > 0:
        depth[depth > cfg.max_depth] = np.nan
    else:
        # Auto-protect: Find the real maximum depth of the valid pixels, discarding extreme edge spikes
        finite_depths = depth[np.isfinite(depth)]
        if finite_depths.size > 0:
            auto_max_limit = np.percentile(finite_depths, 98)
            depth[depth > auto_max_limit] = np.nan

    print("\nMatching + filtering took %.2f s | valid pixels: %.1f%%"
          % (time.time() - t0, 100 * np.isfinite(depth).mean()))

    # ---- save --------------------------------------------------------------
    np.save(os.path.join(args.out_dir, "depth.npy"), depth)
    np.save(os.path.join(args.out_dir, "disparity.npy"), np.where(keep, final, np.nan).astype(np.float32))
    cv2.imwrite(os.path.join(args.out_dir, "left_rectified.png"), l_rect)
    cv2.imwrite(os.path.join(args.out_dir, "right_rectified.png"), r_rect)
    fin = np.isfinite(depth)
    if fin.any():
        lo, hi = np.percentile(depth[fin], [2, 98])
        norm = np.nan_to_num(np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1))
        vis = cv2.applyColorMap((255 * (1 - norm)).astype(np.uint8), cv2.COLORMAP_TURBO)
        vis[~fin] = 0
        cv2.imwrite(os.path.join(args.out_dir, "depth_color.png"), vis)
    if args.save_ply:
        save_ply(os.path.join(args.out_dir, "cloud.ply"), rect["Q"], np.where(keep, final, 0),
                 l_rect, keep & np.isfinite(depth))
    print("Saved results to ./%s/" % args.out_dir)

    show_plots(l_rect, r_rect, raw, depth, max(args.surface_step, 1), args.out_dir, args.clip_pct)
    if not args.no_plot:
        plt.show()


if __name__ == "__main__":
    main()