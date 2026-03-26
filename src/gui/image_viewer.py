"""ImageViewer — QGraphicsView s podporou ROI rubber-band a overlay vizualizáciou."""

from __future__ import annotations

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QRectF, QPointF, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QColor, QPen, QBrush, QPainter
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsLineItem,
    QRubberBand, QSizePolicy,
)

from src.core.roi import ROI


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

        # ROI rubber-band stav
        self._roi_mode = False
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

    def set_roi_mode(self, enabled: bool) -> None:
        """Zapne/vypne režim kreslenia ROI myšou."""
        self._roi_mode = enabled
        if enabled:
            self._view.setDragMode(QGraphicsView.DragMode.NoDrag)
            self._view.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self._view.setCursor(Qt.CursorShape.ArrowCursor)
            if self._rubber_band:
                self._rubber_band.hide()

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
    ) -> None:
        """Vykreslí overlay: Canny hrany + šípka posunutia.

        Args:
            dx_px: posun v x [px]
            dy_px: posun v y [px]
            edges: uint8 grayscale maska hrán (voliteľné)
        """
        self.clear_overlay()

        if self._pixmap_item is None:
            return

        # Canny hrany — cyanová farba
        if edges is not None and edges.ndim == 2:
            h, w = edges.shape
            edge_color = np.zeros((h, w, 4), dtype=np.uint8)
            mask = edges > 0
            edge_color[mask] = [0, 255, 255, 200]  # RGBA cyanová
            q_img = QImage(
                edge_color.data, w, h, w * 4, QImage.Format.Format_RGBA8888
            )
            px_item = self._scene.addPixmap(QPixmap.fromImage(q_img.copy()))
            px_item.setZValue(1)
            self._overlay_items.append(px_item)

        # Šípka posunutia — žltá, zo stredu obrazu
        if self._pixmap_item is not None:
            br = self._pixmap_item.boundingRect()
            cx, cy = br.width() / 2, br.height() / 2
            pen_arrow = QPen(QColor(255, 255, 0))
            pen_arrow.setWidth(3)
            line = self._scene.addLine(cx, cy, cx + dx_px * 5, cy + dy_px * 5, pen_arrow)
            line.setZValue(2)
            self._overlay_items.append(line)

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
        from PyQt6.QtGui import QMouseEvent

        if source is self._view.viewport() and self._roi_mode:
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
                    self.roi_selected.emit(x0, y0, x1, y1)
                return True

        return super().eventFilter(source, event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.fit_in_view()

    # ------------------------------------------------------------------
    # Privátne
    # ------------------------------------------------------------------

    def _remove_roi_item(self) -> None:
        if self._roi_item is not None:
            self._scene.removeItem(self._roi_item)
            self._roi_item = None
