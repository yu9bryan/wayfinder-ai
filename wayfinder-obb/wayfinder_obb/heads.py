"""窄偵測頭（narrow head）patch。

Ultralytics 的偵測頭把 box 回歸分支的寬度綁在 reg_max 上：

    c2 = max(16, ch[0] // 4, reg_max * 4)

reg_max=16 時 c2 固定為 64，不管骨幹多窄都一樣，小模型的參數大多被 head 吃掉。
這裡保留 reg_max=16（輸出幾何與 DFL 解碼完全不變），只把 box 分支寬度直接設成 c2，
讓參數從 head 移回骨幹。

本 repo 的 .pt 權重都是用 patch 後的 head 訓練的，**載入前必須先套用相同的 patch**，
否則 state_dict 形狀對不上。TFLite／ONNX 的計算圖已固化，不需要 patch。
"""
import torch.nn as nn
import ultralytics.nn.tasks as tasks
from ultralytics.nn.modules.conv import Conv, DWConv
from ultralytics.nn.modules.head import OBB, Detect

_ORIG_OBB_INIT = OBB.__init__
_ORIG_DETECT_INIT = Detect.__init__


def _tower(ch, w, out, dw):
    if dw:
        return nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, w, 1)),
                nn.Sequential(DWConv(w, w, 3), Conv(w, w, 1)),
                nn.Conv2d(w, out, 1),
            )
            for x in ch
        )
    return nn.ModuleList(nn.Sequential(Conv(x, w, 3), Conv(w, w, 3), nn.Conv2d(w, out, 1)) for x in ch)


def patch_obb_head(c2=None, c4=None, dw=False):
    """OBB head：box 分支寬度設為 c2；dw=True 時 box／角度分支都改用 depthwise-separable。"""

    def patched(self, nc=80, ne=1, reg_max=16, end2end=False, ch=()):
        _ORIG_OBB_INIT(self, nc, ne, reg_max, end2end, ch)
        if c2 is not None or dw:
            w = c2 if c2 is not None else max(16, ch[0] // 4, self.reg_max * 4)
            self.cv2 = _tower(ch, w, 4 * self.reg_max, dw)
        if c4 is not None or dw:
            w4 = c4 if c4 is not None else max(ch[0] // 4, self.ne)
            self.cv4 = _tower(ch, w4, self.ne, dw)

    tasks.OBB.__init__ = patched


def patch_detect_head(c2=None, dw=False):
    """Detect head：box 分支寬度設為 c2；dw=True 時改用 depthwise-separable。"""

    def patched(self, nc=80, reg_max=16, end2end=False, ch=()):
        _ORIG_DETECT_INIT(self, nc, reg_max, end2end, ch)
        if c2 is None and not dw:
            return
        w = c2 if c2 is not None else max(16, ch[0] // 4, self.reg_max * 4)
        self.cv2 = _tower(ch, w, 4 * self.reg_max, dw)

    tasks.Detect.__init__ = patched
