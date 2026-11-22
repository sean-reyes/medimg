"""Echo ejection fraction (EF) model for the app: outline the left ventricle on two frames and estimate EF.

Needs models/camus_unet_192.pt from train_camus.py (with RES = 192) and: python -m pip install torch pillow scipy numpy
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage

RES = 192
MODEL_PATH = "models/camus_unet_192.pt"

# Placeholder rules. They are sensible guesses, not tuned values: check them against held-out data before trusting them.
MIN_VENTRICLE_FRACTION = 0.01  # the ventricle must cover at least this share of the image
MIN_BLOB_SHARE = 0.90          # at least this share of ventricle pixels must be one connected region
EF_MIN, EF_MAX = 0.10, 0.85    # outside this range the estimate is treated as implausible

ERROR_NOTE = ("On 67 held-out test patients from the same dataset, the typical error was about 4 to 6 percentage "
              "points and 82% of estimates were within 10 points. It can be off by more, especially on images "
              "unlike the training data. This is an approximate value, not a measurement.")


def block(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):  # identical to the one in train_camus.py
    def __init__(self, c=16, classes=4):
        super().__init__()
        self.e1, self.e2, self.e3 = block(1, c), block(c, 2 * c), block(2 * c, 4 * c)
        self.mid = block(4 * c, 8 * c)
        self.u3, self.d3 = nn.ConvTranspose2d(8 * c, 4 * c, 2, stride=2), block(8 * c, 4 * c)
        self.u2, self.d2 = nn.ConvTranspose2d(4 * c, 2 * c, 2, stride=2), block(4 * c, 2 * c)
        self.u1, self.d1 = nn.ConvTranspose2d(2 * c, c, 2, stride=2), block(2 * c, c)
        self.out = nn.Conv2d(c, classes, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        m = self.mid(F.max_pool2d(e3, 2))
        d3 = self.d3(torch.cat([self.u3(m), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.out(d1)


def load_model():
    model = UNet()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


def _prep(image):
    """Grayscale, shrink to RES x RES by averaging, then standardize, as in training."""
    w, h = image.size
    gray = np.asarray(image.convert("L").resize((RES, RES), Image.Resampling.BOX), dtype=np.uint8)
    x = gray.astype(np.float32)
    return gray, (x - x.mean()) / (x.std() + 1e-6), w / h


def _segment(model, x):
    with torch.no_grad():
        return model(torch.from_numpy(x[None, None])).argmax(1)[0].numpy()


def _main_blob_share(mask):
    lab, n = ndimage.label(mask == 1)
    if n == 0:
        return 0.0
    return float(np.max(ndimage.sum(mask == 1, lab, range(1, n + 1))) / (mask == 1).sum())


def disk_volume(mask, n_slices=20):
    """Method of disks along the ventricle's long axis (pixel units; the unit cancels in EF)."""
    ys, xs = np.nonzero(mask == 1)
    if len(ys) < 50:
        return np.nan
    pts = np.stack([xs, ys], 1).astype(float)
    centre = pts.mean(0)
    axis = np.linalg.svd(pts - centre, full_matrices=False)[2][0]
    t = (pts - centre) @ axis
    h = (t.max() - t.min()) / n_slices
    idx = np.minimum(((t - t.min()) / h).astype(int), n_slices - 1)
    width = np.bincount(idx, minlength=n_slices) / h
    return float(np.sum(np.pi / 4 * width ** 2 * h))


def _overlay(gray, mask):
    """Grayscale image with the ventricle tinted red."""
    rgb = np.stack([gray] * 3, -1).astype(float)
    sel = mask == 1
    rgb[sel] = 0.6 * rgb[sel] + 0.4 * np.array([255, 60, 60])
    return rgb.astype(np.uint8)


def analyse(model, ed_image, es_image):
    """Return a dict: status ('ok', 'unsure' or 'reject'), messages, ef (only when ok), and two overlay images."""
    gray_ed, x_ed, r_ed = _prep(ed_image)
    gray_es, x_es, r_es = _prep(es_image)
    m_ed, m_es = _segment(model, x_ed), _segment(model, x_es)

    reject, unsure = [], []
    if not (0.8 <= r_ed <= 1.25 and 0.8 <= r_es <= 1.25):
        reject.append("The images are not roughly square, so they may be cropped or not four-chamber echo frames.")
    for name, m in (("end-of-filling", m_ed), ("end-of-squeezing", m_es)):
        if (m == 1).mean() < MIN_VENTRICLE_FRACTION:
            reject.append(f"No left ventricle was found in the {name} frame.")
        elif _main_blob_share(m) < MIN_BLOB_SHARE:
            unsure.append(f"The ventricle outline in the {name} frame is fragmented.")

    ef = None
    if not reject:
        ef = 1 - disk_volume(m_es) / disk_volume(m_ed)
        if ef < 0:
            unsure.append("The end-of-squeezing ventricle is larger than the end-of-filling one. "
                          "Check that the frames are the right way round.")
        elif ef < EF_MIN or ef > EF_MAX:
            unsure.append(f"The estimated EF ({ef:.0%}) is outside the plausible range.")

    status = "reject" if reject else "unsure" if unsure else "ok"
    return {"status": status, "messages": reject or unsure, "ef": ef if status == "ok" else None,
            "ed_overlay": _overlay(gray_ed, m_ed), "es_overlay": _overlay(gray_es, m_es)}