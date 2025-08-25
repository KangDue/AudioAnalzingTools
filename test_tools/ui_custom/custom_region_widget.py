from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtCore import pyqtSignal

class CustomRegionWidget(QWidget):
    """단순하고 직관적인 영역 위젯. add(widget, name)와 find(name) 제공"""
    signal_emitted = pyqtSignal(str, dict)

    def __init__(self, name=None, parent=None):
        super().__init__(parent)
        self.name = name
        self._widgets = {}
        self._layout = QVBoxLayout(self)
        self.setLayout(self._layout)

    def add(self, widget, name=None):
        """위젯을 레이아웃에 추가하고 필요 시 이름 등록"""
        if name:
            self._widgets[name] = widget
        self._layout.addWidget(widget)
        return widget

    def add_named_widget(self, name, widget):
        return self.add(widget, name)

    def get_widget(self, name):
        return self._widgets.get(name)

    def find(self, name):
        """부모 체인을 거슬러 올라가며 이름으로 검색"""
        if name in self._widgets:
            return self._widgets[name]
        p = self.parent()
        while p is not None:
            if hasattr(p, "get_widget"):
                w = p.get_widget(name)
                if w:
                    return w
            p = p.parent()
        return None

    def set_contents(self, widget):
        while self._layout.count():
            child = self._layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self._layout.addWidget(widget)