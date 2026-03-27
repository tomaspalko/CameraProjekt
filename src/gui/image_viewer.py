"""ImageViewer — QGraphicsView s podporou ROI rubber-band a overlay vizualizáciou."""

from __future__ import annotations

import cv2
import math
from enum import IntEnum

import numpy as np
from PyQt6.QtCore import Qt, QRectF, QPointF, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QColor, QPen, QBrush, QPainter
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsLineItem,
    QRubberBand, QSizePolicy,
)

from src.core.roi import ROI


class ViewerMode(IntEnum):
    """Režim ImageViewera — určuje ako sa interpretujú udalosti myši."""
    NONE         = 0
    ROI          = 1   # rubber-band → emituje roi_selected
    CLICK        = 2   # single klik → emituje image_clicked
    SEGMENT_AREA = 3   # rubber-band → emituje segment_area_selected


def _numpy_to_pixmap(img: np.ndarray) -> QPixmap:
    """Konvertuje numpy obraz (BGR alebo grayscale uint8) na QPixmap."""
    if img.ndim == 2:
        h, w = img.shape
        q_img = QImage(img.data, w, h, w, QImage.Format.Format_Grayscale8)
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        q_img = QImage(img_rgb.data, w, h, w * ch, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(q_img.copy())


class ImageViewer(QWidget):
    """Zobrazovač obrázkov s podporou ROI výberu a overlay vrstiev.

    Signals:
        roi_selected(x0, y0, x1, y1): emitovaný po dokončení ROI výberu myšou.
    """

    roi_selected = pyqtSignal(int, int, int, int)
    image_clicked = pyqtSignal(int, int)            # (x, y) v súradniciach obrazu
    segment_area_selected = pyqtSignal(int, int, int, int)  # (x0, y0, x1, y1) pre výber segmentov

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self._view = QGraphicsView(self._scene, self)
        self._view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._view.setDragMode(QGraphicsView.DragMode.NoDrag)
        self._view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._view)

        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._roi_item: QGraphicsRectItem | None = None
        self._overlay_items: list = []

        # Režim interakcie myšou
        self._mode: ViewerMode = ViewerMode.NONE
        self._rb_origin: QPointF | None = None
        self._rubber_band: QRubberBand | None = None

        self._view.viewport().installEventFilter(self)

    # ------------------------------------------------------------------
    # Verejné metódy
    # ------------------------------------------------------------------

    def set_image(self, img: np.ndarray) -> None:
        """Zobrazí numpy obraz (BGR alebo grayscale)."""
        self._scene.clear()
        self._pixmap_item = None
        self._roi_item = None
        self._overlay_items = []

        px = _numpy_to_pixmap(img)
        self._pixmap_item = self._scene.addPixmap(px)
        self._scene.setSceneRect(QRectF(px.rect()))
        self.fit_in_view()

    def set_mode(self, mode: ViewerMode) -> None:
        """Nastaví aktívny režim interakcie myšou."""
        self._mode = mode
        if mode in (ViewerMode.ROI, ViewerMode.SEGMENT_AREA):
            self._view.setDragMode(QGraphicsView.DragMode.NoDrag)
            self._view.setCursor(Qt.CursorShape.CrossCursor)
        elif mode == ViewerMode.CLICK:
            self._view.setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            self._view.setCursor(Qt.CursorShape.ArrowCursor)
            if self._rubber_band:
                self._rubber_band.hide()

    def set_roi_mode(self, enabled: bool) -> None:
        """Kompatibilný wrapper — zapne/vypne ROI mód."""
        self.set_mode(ViewerMode.ROI if enabled else ViewerMode.NONE)

    def set_click_mode(self, enabled: bool) -> None:
        """Kompatibilný wrapper — zapne/vypne klikací mód."""
        self.set_mode(ViewerMode.CLICK if enabled else ViewerMode.NONE)

    def draw_roi(self, roi: ROI) -> None:
        """Vykreslí prerušovaný obdĺžnik ROI."""
        self._remove_roi_item()
        pen = QPen(QColor(255, 165, 0))  # oranžová
        pen.setWidth(2)
        pen.setStyle(Qt.PenStyle.DashLine)
        rect = QRectF(roi.x0, roi.y0, roi.width, roi.height)
        self._roi_item = self._scene.addRect(rect, pen, QBrush(Qt.BrushStyle.NoBrush))

    def clear_roi(self) -> None:
        """Odstráni vizualizáciu ROI."""
        self._remove_roi_item()

    def draw_overlay(
        self,
        dx_px: float,
        dy_px: float,
        edges: np.ndarray | None = None,
        edge_color: tuple[int, int, int, int] = (0, 255, 255, 200),
    ) -> None:
        """Vykreslí overlay: Canny hrany + šípka posunutia.

        Args:
            dx_px: posun v x [px]
            dy_px: posun v y [px]
            edges: uint8 grayscale maska hrán alebo RGBA array (voliteľné)
            edge_color: RGBA farba hrán keď edges je grayscale (default: cyanová)
        """
        self.clear_overlay()

        if self._pixmap_item is None:
            return

        # Canny hrany — grayscale maska alebo hotové RGBA
        if edges is not None:
            if edges.ndim == 3 and edges.shape[2] == 4:
                arr = np.ascontiguousarray(edges)
                h, w = arr.shape[:2]
                q_img = QImage(
                    arr.data, w, h, w * 4, QImage.Format.Format_RGBA8888
                )
                px_item = self._scene.addPixmap(QPixmap.fromImage(q_img.copy()))
                px_item.setZValue(1)
                self._overlay_items.append(px_item)
            elif edges.ndim == 2:
                h, w = edges.shape
                rgba = np.zeros((h, w, 4), dtype=np.uint8)
                rgba[edges > 0] = edge_color
                q_img = QImage(
                    rgba.data, w, h, w * 4, QImage.Format.Format_RGBA8888
                )
                px_item = self._scene.addPixmap(QPixmap.fromImage(q_img.copy()))
                px_item.setZValue(1)
                self._overlay_items.append(px_item)

        # Šípka posunutia — žltá, zo stredu obrazu (dĺžka capovaná na 40 % min. rozmeru)
        if self._pixmap_item is not None:
            br = self._pixmap_item.boundingRect()
            cx, cy = br.width() / 2, br.height() / 2
            max_arrow = min(br.width(), br.height()) * 0.4
            scale = 5.0
            dx_draw = dx_px * scale
            dy_draw = dy_px * scale
            mag = math.hypot(dx_draw, dy_draw)
            if mag > max_arrow and mag > 0:
                factor = max_arrow / mag
                dx_draw *= factor
                dy_draw *= factor
            pen_arrow = QPen(QColor(255, 255, 0))
            pen_arrow.setWidth(3)
            line = self._scene.addLine(cx, cy, cx + dx_draw, cy + dy_draw, pen_arrow)
            line.setZValue(2)
            self._overlay_items.append(line)

    def draw_edges(self, edges: np.ndarray) -> None:
        """Overlay Canny edge mask (cyan) — without displacement arrow."""
        self.clear_overlay()
        if self._pixmap_item is None or edges is None:
            return
        h, w = edges.shape
        edge_color = np.zeros((h, w, 4), dtype=np.uint8)
        edge_color[edges > 0] = [0, 255, 255, 200]
        q_img = QImage(edge_color.data, w, h, w * 4, QImage.Format.Format_RGBA8888)
        px_item = self._scene.addPixmap(QPixmap.fromImage(q_img.copy()))
        px_item.setZValue(1)
        self._overlay_items.append(px_item)

    def clear_overlay(self) -> None:
        """Odstráni všetky overlay vrstvy (hrany, šípky)."""
        for item in self._overlay_items:
            self._scene.removeItem(item)
        self._overlay_items = []

    def fit_in_view(self) -> None:
        """Prispôsobí zoom tak, aby bol celý obraz viditeľný."""
        if self._pixmap_item is not None:
            self._view.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    # ------------------------------------------------------------------
    # Event filter — rubber-band ROI
    # ------------------------------------------------------------------

    def eventFilter(self, source, event) -> bool:
        from PyQt6.QtCore import QEvent

        if source is not self._view.viewport():
            return super().eventFilter(source, event)

        if self._mode in (ViewerMode.ROI, ViewerMode.SEGMENT_AREA):
            if event.type() == QEvent.Type.MouseButtonPress:
                self._rb_origin = event.position()
                if self._rubber_band is None:
                    self._rubber_band = QRubberBand(
                        QRubberBand.Shape.Rectangle, self._view.viewport()
                    )
                from PyQt6.QtCore import QPoint, QRect
                origin_int = QPoint(int(self._rb_origin.x()), int(self._rb_origin.y()))
                self._rubber_band.setGeometry(QRect(origin_int, origin_int))
                self._rubber_band.show()
                return True

            elif event.type() == QEvent.Type.MouseMove and self._rb_origin is not None:
                from PyQt6.QtCore import QPoint, QRect
                origin_int = QPoint(int(self._rb_origin.x()), int(self._rb_origin.y()))
                curr_int = QPoint(int(event.position().x()), int(event.position().y()))
                self._rubber_band.setGeometry(
                    QRect(origin_int, curr_int).normalized()
                )
                return True

            elif event.type() == QEvent.Type.MouseButtonRelease and self._rb_origin is not None:
                if self._rubber_band:
                    self._rubber_band.hide()

                # Preveď súradnice viewport → scene → obraz
                origin_vp = QPointF(self._rb_origin)
                end_vp = event.position()

                p0 = self._view.mapToScene(int(origin_vp.x()), int(origin_vp.y()))
                p1 = self._view.mapToScene(int(end_vp.x()), int(end_vp.y()))

                x0 = int(min(p0.x(), p1.x()))
                y0 = int(min(p0.y(), p1.y()))
                x1 = int(max(p0.x(), p1.x()))
                y1 = int(max(p0.y(), p1.y()))

                self._rb_origin = None
                if x1 > x0 and y1 > y0:
                    if self._mode == ViewerMode.ROI:
                        self.roi_selected.emit(x0, y0, x1, y1)
                    else:
                        self.segment_area_selected.emit(x0, y0, x1, y1)
                return True

        elif self._mode == ViewerMode.CLICK:
            if event.type() == QEvent.Type.MouseButtonPress:
                vp = event.position()
                sp = self._view.mapToScene(int(vp.x()), int(vp.y()))
                x, y = int(sp.x()), int(sp.y())
                if self._pixmap_item is not None:
                    br = self._pixmap_item.boundingRect()
                    if 0 <= x < int(br.width()) and 0 <= y < int(br.height()):
                        self.image_clicked.emit(x, y)
                return True

        return super().eventFilter(source, event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.fit_in_view()

    # ------------------------------------------------------------------
    # Privátne
    # ------------------------------------------------------------------

    def draw_edges_with_selection(
        self,
        edges: np.ndarray,
        labels: np.ndarray,
        selected_label: int,
    ) -> None:
        """Overlay hrán s výberom: vybraný segment oranžový, ostatné sivé."""
        self.clear_overlay()
        if self._pixmap_item is None or edges is None:
            return
        h, w = edges.shape[:2]
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        # Všetky aktívne hrany → sivé
        rgba[edges > 0] = [80, 80, 80, 140]
        # Vybraný segment → oranžový (prepíše sivú)
        rgba[labels == selected_label] = [255, 165, 0, 240]
        arr = np.ascontiguousarray(rgba)
        q_img = QImage(arr.data, w, h, w * 4, QImage.Format.Format_RGBA8888)
        px_item = self._scene.addPixmap(QPixmap.fromImage(q_img.copy()))
        px_item.setZValue(1)
        self._overlay_items.append(px_item)

    def draw_match_box(self, x: int, y: int, w: int, h: int, score: float) -> None:
        """Nakreslí obdĺžnik výsledku template matchingu + NCC skóre text."""
        color = QColor(0, 255, 0) if score >= 0.5 else QColor(255, 80, 80)
        pen = QPen(color, 2)
        rect = self._scene.addRect(float(x), float(y), float(w), float(h), pen,
                                   QBrush(Qt.BrushStyle.NoBrush))
        rect.setZValue(3)
        self._overlay_items.append(rect)
        txt = self._scene.addSimpleText(f"NCC={score:.2f}")
        txt.setBrush(QBrush(color))
        txt.setPos(float(x), float(y) - 16)
        txt.setZValue(3)
        self._overlay_items.append(txt)

    def draw_match_result(
        self,
        ref_x: int, ref_y: int, w: int, h: int,
        match_x: int, match_y: int,
        score: float,
    ) -> None:
        """Zobrazí výsledok template matchingu:
        - Prerušovaný oranžový box = referenčná poloha (kde BOL segment)
        - Plný zelený/červený box  = nájdená poloha (kde JE)
        - Žltá šípka               = vektor posunutia
        - Text NCC skóre
        """
        # Referenčná poloha — prerušovaný oranžový obdĺžnik
        pen_ref = QPen(QColor(255, 165, 0), 1, Qt.PenStyle.DashLine)
        r_ref = self._scene.addRect(float(ref_x), float(ref_y), float(w), float(h),
                                    pen_ref, QBrush(Qt.BrushStyle.NoBrush))
        r_ref.setZValue(3)
        self._overlay_items.append(r_ref)

        # Nájdená poloha — plný obdĺžnik (zelená ≥ 0.7, žltá ≥ 0.5, červená inak)
        if score >= 0.7:
            found_color = QColor(0, 255, 80)
        elif score >= 0.5:
            found_color = QColor(255, 220, 0)
        else:
            found_color = QColor(255, 80, 80)
        pen_found = QPen(found_color, 2)
        r_found = self._scene.addRect(float(match_x), float(match_y), float(w), float(h),
                                      pen_found, QBrush(Qt.BrushStyle.NoBrush))
        r_found.setZValue(3)
        self._overlay_items.append(r_found)

        # Šípka od stredu ref → stred found
        cx_ref = ref_x + w / 2.0
        cy_ref = ref_y + h / 2.0
        cx_match = match_x + w / 2.0
        cy_match = match_y + h / 2.0
        dx = cx_match - cx_ref
        dy = cy_match - cy_ref
        if abs(dx) > 1 or abs(dy) > 1:
            pen_arr = QPen(QColor(255, 255, 0), 2)
            arrow = self._scene.addLine(cx_ref, cy_ref, cx_match, cy_match, pen_arr)
            arrow.setZValue(4)
            self._overlay_items.append(arrow)

        # Text NCC + dx/dy
        sign_x = "+" if dx >= 0 else ""
        sign_y = "+" if dy >= 0 else ""
        label = f"NCC={score:.2f}  dx={sign_x}{dx:.0f}  dy={sign_y}{dy:.0f}"
        txt = self._scene.addSimpleText(label)
        txt.setBrush(QBrush(found_color))
        txt.setPos(float(match_x), float(match_y) - 18)
        txt.setZValue(4)
        self._overlay_items.append(txt)

    def _remove_roi_item(self) -> None:
        if self._roi_item is not None:
            self._scene.removeItem(self._roi_item)
            self._roi_item = None
