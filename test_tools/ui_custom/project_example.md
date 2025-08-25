from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtCore import pyqtSignal

class CustomRegionWidget(QWidget):
    # 필요 시 signal 정의도 가능
    custom_signal = pyqtSignal(str)

    def __init__(self, name=None, parent=None):
        super().__init__(parent)
        self.name = name
        self.widget_registry = {}
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

    def add_named_widget(self, name, widget):
        self.widget_registry[name] = widget
        self.layout.addWidget(widget)

    def get_widget(self, name):
        return self.widget_registry.get(name, None)

    def set_contents(self, widget):
        """기존 layout 클리어 후 widget 하나만 추가"""
        while self.layout.count():
            child = self.layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.layout.addWidget(widget)

---

# v2: 더 편하고, 직관적이고, 효율적인 레이아웃 설계(보강판)

아래는 기존 설계를 바탕으로 “더 적은 코드로 더 많은 레이아웃을 만들고”, “상태를 자동 저장/복원”하며, “탭/도킹/동적 추가”까지 쉽게 확장할 수 있도록 보강한 내용이야.

## 1) 한 함수로 끝내는 선언적 레이아웃 빌더(Quick Start)

아래처럼 사전(dict) 형태의 스펙만 넘기면 중첩 분할(QSplitter)과 각 영역 위젯(CustomRegionWidget)을 자동으로 생성할 수 있어.

```python
from PyQt5.QtWidgets import QWidget, QSplitter, QVBoxLayout, QLabel, QPushButton
from PyQt5.QtCore import Qt

# spec 예시
spec = {
    "type": "split",              # split | region | tabs(옵션)
    "orientation": "h",          # h | v
    "sizes": [2, 3],              # 초기 비율(옵션)
    "children": [
        {
            "type": "region",
            "name": "left",
            "widgets": [
                ("label", QLabel("왼쪽 패널")),
                ("btn", QPushButton("왼쪽 버튼")),
            ],
        },
        {
            "type": "split",
            "orientation": "v",
            "children": [
                {
                    "type": "region",
                    "name": "right_top",
                    "widgets": [("label", QLabel("오른쪽 위"))],
                },
                {
                    "type": "region",
                    "name": "right_bottom",
                    "widgets": [("label", QLabel("오른쪽 아래"))],
                },
            ],
        },
    ],
}

class LayoutBuilder:
    def __init__(self, region_cls):
        self.region_cls = region_cls
        self.registry = {}  # 모든 region/widget 이름 레지스트리(전역 조회용)

    def build(self, spec):
        node_type = spec.get("type")
        if node_type == "split":
            splitter = QSplitter(Qt.Horizontal if spec.get("orientation") == "h" else Qt.Vertical)
            for child_spec in spec.get("children", []):
                child = self.build(child_spec)
                splitter.addWidget(child)
            # 초기 비율 지정
            sizes = spec.get("sizes")
            if sizes:
                total = sum(sizes)
                splitter.setSizes([int(1000 * s / total) for s in sizes])
            return splitter
        elif node_type == "region":
            region = self.region_cls(spec.get("name"))
            for name, widget in spec.get("widgets", []):
                region.add(name=name, widget=widget)
                self.registry[f"{spec.get('name')}.{name}"] = widget
            self.registry[spec.get("name")] = region
            return region
        elif node_type == "tabs":
            # 아래 3장 TabRegionWidget 참고(간단 구현)
            from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTabWidget
            container = QWidget()
            layout = QVBoxLayout(container)
            tabs = QTabWidget()
            tabs.setMovable(True)  # 탭 드래그로 순서 변경 가능
            layout.addWidget(tabs)
            for tab_spec in spec.get("tabs", []):
                child = self.build(tab_spec["content"])  # content는 split/region 가능
                tabs.addTab(child, tab_spec.get("title", "Untitled"))
            return container
        else:
            raise ValueError(f"Unknown spec type: {node_type}")
```

사용법(예):

```python
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout
import sys

# 위에서 정의한 CustomRegionWidget v2를 사용한다고 가정(아래 2장 참고)
from custom_region_widget import CustomRegionWidget

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("선언적 레이아웃 빌더 예제")
        self.resize(900, 600)
        layout = QVBoxLayout(self)

        builder = LayoutBuilder(CustomRegionWidget)
        root = builder.build(spec)
        layout.addWidget(root)

        # 전역 레지스트리 활용 예: 다른 영역 위젯 쉽게 찾기
        left_btn = builder.registry["left.btn"]
        right_bottom_label = builder.registry["right_bottom.label"]
        left_btn.clicked.connect(lambda: right_bottom_label.setText("왼쪽 버튼에서 변경!"))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = App()
    win.show()
    sys.exit(app.exec_())
```

핵심 요약:
- 분할/탭/영역을 “사전 스펙”으로 선언 → 코드량 대폭 절감, 가독성↑
- 전역 레지스트리로 위젯 참조를 단순화 → 상호 통신이 직관적

---

## 2) CustomRegionWidget v2: 더 단순한 API + 계층 검색

```python
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtCore import pyqtSignal

class CustomRegionWidget(QWidget):
    # 영역 단위 이벤트 (필요 시 사용)
    signal_emitted = pyqtSignal(str, dict)

    def __init__(self, name=None, parent=None):
        super().__init__(parent)
        self.name = name
        self._widgets = {}
        self._layout = QVBoxLayout(self)

    # 더 직관적인 add API
    def add(self, widget, name=None):
        if name:
            self._widgets[name] = widget
        self._layout.addWidget(widget)
        return widget

    # 기존 명명 추가 API와 호환
    def add_named_widget(self, name, widget):
        return self.add(widget, name)

    def get_widget(self, name):
        return self._widgets.get(name)

    def find(self, name):
        """부모 체인을 따라 올라가며 동일한 이름을 검색(전역 탐색 대용)."""
        # 현재 영역에서 먼저 찾고
        if name in self._widgets:
            return self._widgets[name]
        # 부모 위젯들 순회하며 이름 속성 검사
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
```

장점:
- add(widget, name) 형태로 직관적인 추가
- find(name)로 상위까지 포함한 간단 검색 → 지역/전역 참조 문제를 간단화
- 필요한 경우 signal_emitted로 느슨 결합 이벤트 처리 가능

---

## 3) 상태 저장/복원 자동화(QSettings + QSplitter.saveState)

앱 종료 시 분할 비율/탭 상태를 자동 저장, 다음 실행에 자동 복원되도록 구성할 수 있어.

```python
from PyQt5.QtCore import QSettings
from PyQt5.QtWidgets import QSplitter, QTabWidget

class LayoutState:
    def __init__(self, org="MyOrg", app="MyApp"):
        self.settings = QSettings(org, app)

    def save(self, root, key="root"):
        # QSplitter 상태 저장
        if isinstance(root, QSplitter):
            self.settings.setValue(f"{key}/splitter", root.saveState())
            for i in range(root.count()):
                self.save(root.widget(i), f"{key}/{i}")
        # QTabWidget 상태 저장(현재 인덱스)
        elif isinstance(root, QTabWidget):
            self.settings.setValue(f"{key}/tab_index", root.currentIndex())
            for i in range(root.count()):
                self.save(root.widget(i), f"{key}/tab/{i}")
        # 일반 위젯(리프)은 패스

    def restore(self, root, key="root"):
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
```

사용 팁:
- 메인 윈도우 closeEvent에서 save, 초기화 시점(예: show 직후)에 restore 호출
- Organization/Application 이름을 고정하면 어디서 실행해도 동일한 키로 저장됨

---

## 4) 탭/도킹 강화: QTabWidget, QMainWindow + QDockWidget

- 탭: QTabWidget.setMovable(True)로 드래그 정렬, setTabsClosable(True)로 닫기 버튼 지원
- 도킹: 복잡한 IDE 스타일은 QMainWindow의 QDockWidget이 훨씬 편리

도킹 예시(핵심만):

```python
from PyQt5.QtWidgets import QMainWindow, QDockWidget, QTextEdit, QWidget

class DockExample(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("도킹 레이아웃 예제")

        # 센터 위젯
        editor = QTextEdit()
        self.setCentralWidget(editor)

        # 도킹 패널들
        project = QDockWidget("Project")
        project.setWidget(QTextEdit("프로젝트 뷰"))
        self.addDockWidget(Qt.LeftDockWidgetArea, project)

        console = QDockWidget("Console")
        console.setWidget(QTextEdit("콘솔"))
        self.addDockWidget(Qt.BottomDockWidgetArea, console)
```

언제 도킹을 고려할까?
- 사용자가 패널을 자유롭게 떼고 붙이며 배치하길 원할 때
- 복잡한 전문가용 UI(IDE, 분석툴 등)에서 높은 유연성이 필요할 때

---

## 5) 동적 영역 추가/삭제 패턴(런타임 구성 변화)

선언적 빌더를 사용할 때도 런타임에 위젯을 추가/제거하는 상황이 잦아. 대표 패턴은 다음과 같아.

```python
# 1) 특정 Region에 새로운 위젯 추가
region = builder.registry["right_top"]  # 빌더 생성 시 등록됨
region.add(QPushButton("동적 버튼"), name="dyn_btn")

# 2) Splitter에 새로운 Region 끼워넣기(단, 구조 제약 고려)
splitter = region.parent()  # 부모가 Splitter라고 가정
new_region = CustomRegionWidget("extra")
splitter.addWidget(new_region)

# 3) 위젯 제거
btn = region.get_widget("dyn_btn")
btn.setParent(None)
btn.deleteLater()
```

주의:
- Splitter 자식 순서가 레이아웃에 영향을 주므로, 추가 시 적절한 index 사용(addWidget 대신 insertWidget도 가능)
- 전역 레지스트리 사용 시, 제거/이름 변경에 맞춰 동기화 필요

---

## 6) 자주 쓰는 레이아웃 템플릿(복붙용)

- 2분할(좌/우): {type: split, orientation: 'h', children: [region, region]}
- 2분할(상/하): {type: split, orientation: 'v', children: [region, region]}
- 3분할(좌/중/우): split(h) + 가운데를 다시 split(v/h)
- IDE 스타일: QMainWindow + Dock 조합(패널 도킹/분리/부동 창)

샘플 스펙(3분할 좌/우+우측 상/하):

```python
spec = {
    "type": "split",
    "orientation": "h",
    "sizes": [1, 2],
    "children": [
        {"type": "region", "name": "left", "widgets": [("label", QLabel("좌"))]},
        {
            "type": "split",
            "orientation": "v",
            "children": [
                {"type": "region", "name": "right_top", "widgets": [("label", QLabel("우상"))]},
                {"type": "region", "name": "right_bottom", "widgets": [("label", QLabel("우하"))]},
            ],
        },
    ],
}
```

---

## 7) 베스트 프랙티스

- 이름 규칙: regionName.widgetName 형태로 전역 레지스트리에 저장 → 충돌 방지, 참조 용이
- 신호 설계: 지역 신호는 Region에서, 전역 브로드캐스트는 중앙 Event(옵션)로 분리
- 성능: 무거운 위젯은 지연 생성(처음 열릴 때 생성/로드)
- 접근성: 라벨과 버튼 텍스트는 i18n을 고려하여 상수/리소스로 분리
- 유지보수: 레이아웃 스펙은 별도 파일(json/yaml)로 분리하는 것도 좋음(파싱만 하면 됨)

---

## 8) 파일 구조 제안(실전 적용용)

```plaintext
/your_project
├── main.py
├── custom_region_widget.py         # CustomRegionWidget v2
├── layout_builder.py               # LayoutBuilder + 상태 저장/복원 유틸
├── examples/
│   ├── ex_quickstart.py            # 선언적 스펙으로 바로 구동
│   ├── ex_tabs.py                  # 탭 혼합 예시
│   └── ex_dock.py                  # 도킹 예시(QMainWindow)
└── resources/
    └── i18n.py                     # 문자열/리소스 모음(옵션)
```

---

## 9) 다음 단계 제안

- 위 파일 구조로 최소 실행 예제를 바로 뽑아드릴 수 있어. 실제 파이썬 파일로 뼈대 생성 + 실행 스크립트까지 원하면 말씀만 해줘.
- 상태 저장/복원 유틸과 레이아웃 빌더는 합쳐서도 가능. 원하시면 프로젝트에 맞춰 API를 더 간결하게 커스터마이즈해줄게.
- 도입 라이브러리는 PyQt5만 사용(추가 외부 라이브러리 없음). 다른 GUI 스택(PySide6, Qt6)으로도 이식 가능.
