from PyQt5.QtWidgets import QSplitter, QWidget, QVBoxLayout, QTabWidget, QMainWindow, QDockWidget, QPushButton, QLabel, QTextEdit
from PyQt5.QtCore import Qt, QSettings
from typing import Dict, Any

class LayoutState:
    """QSplitter/QTabWidget/QMainWindow 상태 저장 및 복원 유틸"""
    def __init__(self, org: str = "MyOrg", app: str = "MyApp"):
        self.settings = QSettings(org, app)

    def save(self, root: QWidget, key: str = "root") -> None:
        if isinstance(root, QSplitter):
            self.settings.setValue(f"{key}/splitter", root.saveState())
            for i in range(root.count()):
                self.save(root.widget(i), f"{key}/{i}")
        elif isinstance(root, QTabWidget):
            self.settings.setValue(f"{key}/tab_index", root.currentIndex())
            for i in range(root.count()):
                self.save(root.widget(i), f"{key}/tab/{i}")
        elif isinstance(root, QMainWindow):
            # QMainWindow 전체 상태 저장(geometry/state)
            self.settings.setValue(f"{key}/main_geometry", root.saveGeometry())
            self.settings.setValue(f"{key}/main_state", root.saveState())
            # 중앙 영역과 각 도킹 내부 컨텐츠도 개별 저장
            if root.centralWidget() is not None:
                self.save(root.centralWidget(), f"{key}/central")
            for dock in root.findChildren(QDockWidget):
                name = dock.objectName() or "dock"
                if dock.widget() is not None:
                    self.save(dock.widget(), f"{key}/dock/{name}")
        # 기타 위젯은 저장하지 않음

    def restore(self, root: QWidget, key: str = "root") -> None:
        if isinstance(root, QSplitter):
            state = self.settings.value(f"{key}/splitter")
            if state is not None:
                root.restoreState(state)
            for i in range(root.count()):
                self.restore(root.widget(i), f"{key}/{i}")
        elif isinstance(root, QTabWidget):
            idx = self.settings.value(f"{key}/tab_index")
            if idx is not None:
                try:
                    root.setCurrentIndex(int(idx))
                except Exception:
                    pass
            for i in range(root.count()):
                self.restore(root.widget(i), f"{key}/tab/{i}")
        elif isinstance(root, QMainWindow):
            # 도킹/중앙 위젯들이 이미 생성된 이후에 복원해야 함
            geom = self.settings.value(f"{key}/main_geometry")
            if geom is not None:
                root.restoreGeometry(geom)
            st = self.settings.value(f"{key}/main_state")
            if st is not None:
                root.restoreState(st)
            if root.centralWidget() is not None:
                self.restore(root.centralWidget(), f"{key}/central")
            for dock in root.findChildren(QDockWidget):
                name = dock.objectName() or "dock"
                if dock.widget() is not None:
                    self.restore(dock.widget(), f"{key}/dock/{name}")


class LayoutBuilder:
    """사전(spec) 기반으로 Split/Region/Tabs/MainWindow+Dock를 생성하는 빌더"""
    def __init__(self, region_cls, widget_factories: Dict[str, Any] | None = None):
        self.region_cls = region_cls
        self.registry: Dict[str, Any] = {}
        # 재사용성 강화를 위한 기본 위젯 팩토리 레지스트리(지연 생성 불필요: QtApp 이후 build 호출됨)
        self.widget_factories: Dict[str, Any] = {
            "label": lambda text="": QLabel(text),
            "button": lambda text="": QPushButton(text),
            "text_edit": lambda text="": QTextEdit(text),
        }
        if widget_factories:
            self.widget_factories.update(widget_factories)

    def _area_from_str(self, s: str) -> Qt.DockWidgetArea:
        m = {
            "left": Qt.LeftDockWidgetArea,
            "right": Qt.RightDockWidgetArea,
            "top": Qt.TopDockWidgetArea,
            "bottom": Qt.BottomDockWidgetArea,
        }
        return m.get(s.lower(), Qt.LeftDockWidgetArea)

    def _areas_from_list(self, lst) -> Qt.DockWidgetAreas:
        if not lst:
            return Qt.AllDockWidgetAreas
        areas = Qt.NoDockWidgetArea
        for item in lst:
            areas |= self._area_from_str(item)
        return areas

    def _features_from_list(self, lst) -> QDockWidget.DockWidgetFeatures:
        if not lst:
            return QDockWidget.AllDockWidgetFeatures
        mapping = {
            "closable": QDockWidget.DockWidgetClosable,
            "floatable": QDockWidget.DockWidgetFloatable,
            "movable": QDockWidget.DockWidgetMovable,
        }
        feats = QDockWidget.NoDockWidgetFeatures
        for item in lst:
            key = str(item).lower()
            if key in mapping:
                feats |= mapping[key]
        return feats

    def _build_widget_entry(self, entry: Any):
        # entry 형태: (wname, widget|callable) 또는 {name, factory, args, kwargs}
        if isinstance(entry, dict):
            wname = entry.get("name")
            factory = entry.get("factory")
            args = entry.get("args", [])
            kwargs = entry.get("kwargs", {})
            if factory not in self.widget_factories:
                raise KeyError(f"Unknown widget factory: {factory}")
            widget = self.widget_factories[factory](*args, **kwargs)
            return wname, widget
        else:
            # tuple/list 처리
            try:
                wname, widget = entry
            except Exception as e:
                raise ValueError(f"Invalid widget spec entry: {entry}") from e
            if callable(widget):
                widget = widget()
            return wname, widget

    def build(self, spec: Dict[str, Any]) -> QWidget:
        node_type = spec.get("type")
        if node_type == "split":
            splitter = QSplitter(Qt.Horizontal if spec.get("orientation") == "h" else Qt.Vertical)
            for child_spec in spec.get("children", []):
                child = self.build(child_spec)
                splitter.addWidget(child)
            sizes = spec.get("sizes")
            if sizes:
                total = sum(sizes)
                splitter.setSizes([int(1000 * s / total) for s in sizes])
            return splitter
        elif node_type == "region":
            name = spec.get("name")
            region = self.region_cls(name)
            for entry in spec.get("widgets", []):
                wname, w = self._build_widget_entry(entry)
                region.add(w, name=wname)
                if name and wname:
                    self.registry[f"{name}.{wname}"] = w
            if name:
                self.registry[name] = region
            return region
        elif node_type == "tabs":
            container = QWidget()
            layout = QVBoxLayout(container)
            tabs = QTabWidget()
            tabs.setMovable(True)
            layout.addWidget(tabs)
            for tab_spec in spec.get("tabs", []):
                child = self.build(tab_spec["content"])  # content는 split/region 가능
                tabs.addTab(child, tab_spec.get("title", "Untitled"))
            return container
        elif node_type == "main_window":
            mw = QMainWindow()
            if spec.get("title"):
                mw.setWindowTitle(spec.get("title"))
            # central
            central_spec = spec.get("central")
            if central_spec:
                central = self.build(central_spec)
                mw.setCentralWidget(central)
                self.registry["central"] = central
            # docks
            docks_def = spec.get("docks", [])
            docks_map: Dict[str, QDockWidget] = {}
            for dspec in docks_def:
                name = dspec.get("name")
                title = dspec.get("title", name or "Dock")
                area = self._area_from_str(dspec.get("area", "left"))
                content = self.build(dspec.get("content", {"type": "region", "name": name or "dock"}))
                dock = QDockWidget(title)
                if name:
                    dock.setObjectName(name)
                dock.setAllowedAreas(self._areas_from_list(dspec.get("allowedAreas")))
                dock.setFeatures(self._features_from_list(dspec.get("features")))
                dock.setWidget(content)
                mw.addDockWidget(area, dock)
                if dspec.get("floating"):
                    dock.setFloating(True)
                if dspec.get("visible") is False:
                    dock.hide()
                # registry 등록
                if name:
                    docks_map[name] = dock
                    self.registry[f"dock:{name}"] = dock
            # tabify 처리(모든 dock 생성 이후)
            for dspec in docks_def:
                name = dspec.get("name")
                if not name:
                    continue
                for target in dspec.get("tabify_with", []) or []:
                    if name in docks_map and target in docks_map:
                        mw.tabifyDockWidget(docks_map[name], docks_map[target])
            self.registry["main_window"] = mw
            return mw
        else:
            raise ValueError(f"Unknown spec type: {node_type}")