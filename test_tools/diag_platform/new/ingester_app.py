import sys
import os
import re
import orjson
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from backend import DataManager
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import freeze_support

# ==========================================
# [수정됨] 파싱 실패 시 0.0 채우기 (데이터 밀림 방지)
# ==========================================
def process_single_file_task(file_info):
    """
    파일 하나를 읽어서 파싱한 뒤, DB에 넣을 준비가 된 데이터(Dict, Numpy)를 반환합니다.
    """
    sw_path, target_channels, metrics, bands = file_info
    
    try:
        filename = os.path.basename(sw_path)
        pattern = re.compile(r"(\d{8})_(\d{6})_([A-Za-z0-9]+)")
        match = pattern.search(filename)
        if not match: return None

        date_part, time_part, serial_part = match.groups()
        unique_id = f"{date_part}_{time_part}_{serial_part}"

        # 1. SW JSON 읽기
        with open(sw_path, 'rb') as f: 
            sw_json = orjson.loads(f.read())
            
        sr = sw_json.get("SamplesPerSecond", 16000)
        ts_data = sw_json.get("ts", {})
        
        raw_data = {}
        for k, v in ts_data.items():
            if k.startswith("ch_"):
                raw_data[k] = np.array(v, dtype=np.float32)
        
        if not raw_data: return None

        # 2. SF JSON 읽기
        sf_path = sw_path.replace("_SW_", "_SF_")
        feats_by_ch = {}
        
        if os.path.exists(sf_path):
            with open(sf_path, 'rb') as f: 
                sf_json = orjson.loads(f.read())
            for k, v in sf_json.items():
                if "Feature" in k:
                    ch_num = re.search(r"Ch(\d+)", k)
                    if ch_num:
                        ch_key = f"ch_{ch_num.group(1)}"
                        # 내부 파싱 로직
                        parsed = []
                        try:
                            pairs = v.split('&')
                            for pair in pairs:
                                if '=' in pair:
                                    k_in, v_in = pair.split('=', 1)
                                    try: 
                                        parsed.append(float(v_in))
                                    except: 
                                        # [수정] 실패 시 pass가 아니라 0.0을 넣어서 자리를 지킴
                                        parsed.append(0.0) 
                        except: pass
                        feats_by_ch[ch_key] = parsed

        # 3. Feature Flattening
        flat_values = []
        
        for ch in target_channels:
            ch_feats = feats_by_ch.get(ch, [])
            # SF 파일 부재로 리스트가 비어있다면, 0.0으로 49개 채움 (SPL 1개 + 8*6개 = 49개 가정)
            if not ch_feats:
                flat_values.extend([0.0] * 49) 
            else:
                flat_values.extend(ch_feats)

        feat_array = np.array(flat_values, dtype=np.float32)
        
        # Duration
        first_key = next(iter(raw_data))
        duration = len(raw_data[first_key]) / sr
        
        meta = {
            'id': unique_id, 'filename': filename, 'path': sw_path,
            'year': int(date_part[:4]), 'month': int(date_part[4:6]), 'day': int(date_part[6:8]),
            'serial': serial_part, 'duration': duration, 'sr': sr
        }
        
        return (meta, raw_data, feat_array)

    except Exception as e:
        return None


# --- Worker Thread ---
class IngestionWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal()
    
    def __init__(self, folder_paths, db_manager):
        super().__init__()
        self.folder_paths = folder_paths
        self.db = db_manager
        self.batch_size = 500

    def run(self):
        self.progress.emit("🔍 Scanning folders...")
        sw_files = []
        for root_path in self.folder_paths:
            for root, dirs, files in os.walk(root_path):
                for f in files:
                    if ("_SW_" in f or "_sw_" in f) and f.endswith(".json"):
                        sw_files.append(os.path.join(root, f))
        
        total = len(sw_files)
        self.progress.emit(f"📂 Found {total} SW files. Loading DB index...")
        
        existing_ids = self.db.get_all_existing_ids()
        self.progress.emit(f"ℹ️ {len(existing_ids)} existing records loaded.")

        tasks = []
        target_channels = ['ch_1', 'ch_2', 'ch_3', 'ch_4']
        metrics = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6']
        pattern = re.compile(r"(\d{8})_(\d{6})_([A-Za-z0-9]+)")

        skipped_count = 0
        for fpath in sw_files:
            fname = os.path.basename(fpath)
            match = pattern.search(fname)
            if match:
                d, t, s = match.groups()
                uid = f"{d}_{t}_{s}"
                if uid in existing_ids:
                    skipped_count += 1
                    continue
                tasks.append((fpath, target_channels, metrics, bands))
        
        self.progress.emit(f"⚡ Starting Multiprocessing Pool... (To Process: {len(tasks)}, Skipped: {skipped_count})")

        batch_records = []
        processed_count = 0
        
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(process_single_file_task, task): task for task in tasks}
            
            for future in as_completed(futures):
                result = future.result()
                
                if result:
                    batch_records.append(result)
                    processed_count += 1
                    
                    if len(batch_records) >= self.batch_size:
                        if self.db.insert_batch_records(batch_records):
                            self.progress.emit(f"💾 Saved Batch: {processed_count}/{len(tasks)}")
                            batch_records = []
                        else:
                            self.progress.emit("❌ DB Write Failed!")
        
        if batch_records:
            self.db.insert_batch_records(batch_records)
            
        self.progress.emit(f"✅ All Done. Processed: {processed_count}, Skipped: {skipped_count}")
        self.finished.emit()

# --- Main App ---
class IngesterApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Data Ingester (데이터 변환 - MultiCore)")
        self.resize(800, 600)
        self.db_manager = DataManager()
        
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #2b2b2b; color: #ffffff; font-size: 14px; }
            QGroupBox { font-weight: bold; border: 1px solid #555; margin-top: 10px; padding-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }
            QListWidget { background-color: #1e1e1e; color: #fff; border: 1px solid #555; }
            QPushButton { background-color: #555; color: #fff; border: 1px solid #777; border-radius: 3px; padding: 6px; }
            QPushButton:hover { background-color: #666; }
            QTextEdit { background-color: #1e1e1e; color: #fff; border: 1px solid #555; }
            QMessageBox { background-color: #2b2b2b; }
            QMessageBox QLabel { color: #ffffff; background-color: transparent; }
        """)

        self.central = QWidget()
        self.setCentralWidget(self.central)
        self.layout = QVBoxLayout(self.central)
        self.setAcceptDrops(True)

        btn_group = QGroupBox("폴더 관리")
        btn_layout = QHBoxLayout()
        self.btn_add = QPushButton("📂 폴더 추가 (다중)")
        self.btn_add.clicked.connect(self.add_folders_multi)
        self.btn_expand = QPushButton("하위 폴더 펼치기")
        self.btn_expand.clicked.connect(self.expand_subfolders)
        self.btn_remove = QPushButton("🗑️ 선택 제거")
        self.btn_remove.clicked.connect(self.remove_selection)
        self.btn_clear = QPushButton("초기화")
        self.btn_clear.clicked.connect(self.clear_all)
        
        btn_layout.addWidget(self.btn_add); btn_layout.addWidget(self.btn_expand)
        btn_layout.addWidget(self.btn_remove); btn_layout.addWidget(self.btn_clear)
        btn_group.setLayout(btn_layout)
        self.layout.addWidget(btn_group)

        self.layout.addWidget(QLabel("대상 폴더 목록 (Drag & Drop 가능):"))
        self.list_targets = QListWidget()
        self.list_targets.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.layout.addWidget(self.list_targets)

        self.btn_run = QPushButton("🚀 데이터 변환 시작 (Start Ingestion)")
        self.btn_run.setFixedHeight(50)
        self.btn_run.setStyleSheet("font-weight: bold; font-size: 15px; background-color: #007ACC; color: white;")
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self.run_ingestion)
        self.layout.addWidget(self.btn_run)
        
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.layout.addWidget(self.log_view)

    def add_folders_multi(self):
        dlg = QFileDialog(self)
        dlg.setFileMode(QFileDialog.Directory)
        dlg.setOption(QFileDialog.DontUseNativeDialog, True)
        for view in dlg.findChildren(QAbstractItemView): view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        if dlg.exec_(): self.add_paths_to_list(dlg.selectedFiles())

    def add_paths_to_list(self, paths):
        added_count = 0
        existing = set(self.list_targets.item(i).text() for i in range(self.list_targets.count()))
        for path in paths:
            path = os.path.normpath(path)
            if path not in existing and os.path.isdir(path):
                self.list_targets.addItem(path)
                added_count += 1
        
        if added_count > 0: self.check_ready_status()

    def expand_subfolders(self):
        count = self.list_targets.count()
        if count == 0: return
        new_paths = []; remove_indices = []
        for i in range(count):
            parent = self.list_targets.item(i).text()
            try:
                subs = [os.path.join(parent, d) for d in os.listdir(parent) if os.path.isdir(os.path.join(parent, d))]
                if subs: new_paths.extend(subs); remove_indices.append(i)
            except: pass
        
        for i in sorted(remove_indices, reverse=True): self.list_targets.takeItem(i)
        self.add_paths_to_list(new_paths)

    def remove_selection(self):
        for item in self.list_targets.selectedItems(): self.list_targets.takeItem(self.list_targets.row(item))
        self.check_ready_status()

    def clear_all(self):
        self.list_targets.clear()
        self.db_manager.close_db()
        self.check_ready_status()
        self.log_view.clear()

    def check_ready_status(self):
        cnt = self.list_targets.count()
        self.btn_run.setEnabled(cnt > 0)
        self.btn_run.setText(f"🚀 데이터 변환 시작 (대상: {cnt}개)" if cnt > 0 else "🚀 데이터 변환 시작")

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls(): e.accept()
        else: e.ignore()

    def dropEvent(self, e):
        files = [u.toLocalFile() for u in e.mimeData().urls() if os.path.isdir(u.toLocalFile())]
        if files: self.add_paths_to_list(files)

    def run_ingestion(self):
        if not self.db_manager.conn:
            default_dir = self.list_targets.item(0).text() if self.list_targets.count() > 0 else ""
            
            # [수정] 윈도우 기본 탐색기(Native Dialog) 강제 사용
            # options 인자를 명시적으로 넘겨주면 OS 기본 대화상자를 우선 사용합니다.
            options = QFileDialog.Options()
            
            fname, _ = QFileDialog.getSaveFileName(
                self, 
                "출력 DB/H5 저장 (파일명을 입력하세요)", 
                default_dir, 
                "SQLite DB (*.db)", 
                options=options
            )
            
            if not fname: return
                
            self.db_manager.connect_db(fname)
            self.log_view.append(f"💾 DB 저장 위치: {fname}")
            self.log_view.append(f"💾 H5 저장 위치: {fname.replace('.db', '.h5')}")

        targets = [self.list_targets.item(i).text() for i in range(self.list_targets.count())]
        self.btn_run.setEnabled(False)
        self.worker = IngestionWorker(targets, self.db_manager)
        self.worker.progress.connect(self.log_view.append)
        self.worker.finished.connect(lambda: self.check_ready_status())
        self.worker.start()

if __name__ == "__main__":
    freeze_support()
    app = QApplication(sys.argv)
    window = IngesterApp()
    window.show()
    sys.exit(app.exec_())