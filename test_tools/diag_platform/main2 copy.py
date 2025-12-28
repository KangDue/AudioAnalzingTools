import sys
import os
import sqlite3
import h5py
import numpy as np
import orjson
import sounddevice as sd
from scipy import signal
from PyQt5.QtWidgets import *
from PyQt5.QtCore import*
from PyQt5.QtGui import QColor
import pyqtgraph as pg
import pandas as pd
import re
import sounddevice as sd # 소리 재생용

# ==========================================
# 1. Backend & Data Management Class
# ==========================================
class DataManager:
    def __init__(self):
        self.db_path = ""
        self.h5_path = ""
        self.conn = None
        self.cursor = None

    def generate_fixed_names(self):
        channels = ['ch_1', 'ch_2', 'ch_3', 'ch_4']
        metrics = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6']
        names = []
        for ch in channels:
            names.append(f"{ch}_SPL")
            for m in metrics:
                for b in bands:
                    names.append(f"{ch}_{m}_{b}")
        return names

    def connect_db(self, target_path):
        if os.path.isdir(target_path):
            self.db_path = os.path.join(target_path, "factory_noise.db")
        else:
            self.db_path = target_path
            
        base_name = os.path.splitext(self.db_path)[0]
        self.h5_path = f"{base_name}.h5"
        
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        self.cursor.execute("PRAGMA journal_mode=WAL;")
        
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='noise_data';")
        if self.cursor.fetchone():
            self.cursor.execute("PRAGMA table_info(noise_data)")
            cols = [c[1] for c in self.cursor.fetchall()]
            if 'year' not in cols:
                self.cursor.execute("DROP TABLE noise_data"); self.conn.commit()

        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS noise_data (
                id TEXT PRIMARY KEY, filename TEXT, file_path TEXT,
                year INTEGER, month INTEGER, day INTEGER, serial TEXT,
                duration REAL, sample_rate INTEGER,
                label_top TEXT DEFAULT NULL, label_mid TEXT DEFAULT NULL, label_bot TEXT DEFAULT NULL,
                is_labeled INTEGER DEFAULT 0, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS label_settings (
                category TEXT, name TEXT, ord INTEGER,
                PRIMARY KEY(category, name)
            )
        """)

        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_date ON noise_data (year, month, day);")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_serial ON noise_data (serial);")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_label_top ON noise_data (label_top);")
        self.conn.commit()

    def remove_label_from_records(self, col_name, label_value):
        try:
            query = f"UPDATE noise_data SET {col_name} = NULL WHERE {col_name} = ?"
            self.cursor.execute(query, (label_value,))
            self.cursor.execute("UPDATE noise_data SET is_labeled = 0 WHERE label_top IS NULL")
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Delete Label Error: {e}")
            return False

    def fetch_label_settings(self, category):
        try:
            self.cursor.execute("SELECT name FROM label_settings WHERE category = ? ORDER BY ord ASC", (category,))
            return [r['name'] for r in self.cursor.fetchall()]
        except: return []

    def save_label_settings(self, category, label_list):
        try:
            self.cursor.execute("DELETE FROM label_settings WHERE category = ?", (category,))
            data = [(category, name, idx) for idx, name in enumerate(label_list)]
            self.cursor.executemany("INSERT INTO label_settings (category, name, ord) VALUES (?, ?, ?)", data)
            self.conn.commit()
        except Exception as e: print(f"Save Label Config Error: {e}")

    def sync_missing_labels(self):
        categories = {'label_top': 'top', 'label_mid': 'mid', 'label_bot': 'bot'}
        for col, cat in categories.items():
            self.cursor.execute(f"SELECT DISTINCT {col} FROM noise_data WHERE {col} IS NOT NULL")
            used_labels = set([r[0] for r in self.cursor.fetchall()])
            current_labels = set(self.fetch_label_settings(cat))
            missing = used_labels - current_labels
            if missing:
                start_ord = 1000
                data = [(cat, lbl, start_ord + i) for i, lbl in enumerate(missing)]
                self.cursor.executemany("INSERT INTO label_settings (category, name, ord) VALUES (?, ?, ?)", data)
        self.conn.commit()

    def update_single_label(self, uid, col_name, value):
        try:
            if not value or value.strip() == "": value = None
            query = f"UPDATE noise_data SET {col_name} = ? WHERE id = ?"
            self.cursor.execute(query, (value, uid))
            self.cursor.execute("UPDATE noise_data SET is_labeled = CASE WHEN label_top IS NOT NULL AND label_top != '' THEN 1 ELSE 0 END WHERE id = ?", (uid,))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Single Update Error: {e}")
            return False

    def ensure_label_exists(self, category, label_name):
        if not label_name: return False
        try:
            self.cursor.execute("SELECT 1 FROM label_settings WHERE category=? AND name=?", (category, label_name))
            if not self.cursor.fetchone():
                self.cursor.execute("SELECT MAX(ord) FROM label_settings WHERE category=?", (category,))
                res = self.cursor.fetchone()
                max_ord = res[0] if res[0] is not None else 0
                self.cursor.execute("INSERT INTO label_settings (category, name, ord) VALUES (?, ?, ?)", (category, label_name, max_ord + 1))
                self.conn.commit()
                return True
            return False
        except: return False

    def insert_record(self, meta, raw_data, feat_array):
        try:
            self.cursor.execute("""
                INSERT INTO noise_data (id, filename, file_path, year, month, day, serial, duration, sample_rate)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (meta['id'], meta['filename'], meta['path'], 
                  meta['year'], meta['month'], meta['day'], meta['serial'],
                  meta['duration'], meta['sr']))
            
            with h5py.File(self.h5_path, 'a') as h5f:
                if 'global_feat_names' not in h5f.attrs:
                    # IngestionWorker는 feat_names를 안 넘기므로 여기선 저장 안함 (나중에 자동생성)
                    pass 
                
                if meta['id'] in h5f: del h5f[meta['id']]
                grp = h5f.create_group(meta['id'])
                raw_grp = grp.create_group('raw')
                for ch, data in raw_data.items():
                    raw_grp.create_dataset(ch, data=data, compression="gzip", compression_opts=4)
                grp.create_dataset('feature', data=feat_array)
            
            self.conn.commit()
            return True
        except Exception:
            if self.conn: self.conn.rollback()
            return False

    def get_signal_data(self, unique_id):
        try:
            with h5py.File(self.h5_path, 'r') as h5f:
                if unique_id not in h5f: return None, None, None
                raw_grp = h5f[unique_id]['raw']
                raw_data = {k: raw_grp[k][:] for k in raw_grp.keys()}
                feat_array = np.array([])
                feat_names = []
                if 'feature' in h5f[unique_id]:
                    feat_array = h5f[unique_id]['feature'][:]
                    feat_names = self.generate_fixed_names()
                return raw_data, feat_array, feat_names
        except Exception:
            return None, None, None

    def export_training_data(self, output_path):
        self.cursor.execute("SELECT id, label_bot, label_mid, label_top FROM noise_data WHERE is_labeled=1")
        rows = self.cursor.fetchall()
        if not rows: raise Exception("라벨링된 데이터가 없습니다.")

        fixed_names = self.generate_fixed_names()
        export_data = []
        if os.path.exists(self.h5_path):
            with h5py.File(self.h5_path, 'r') as h5f:
                for row in rows:
                    uid = row['id']
                    if uid not in h5f: continue
                    if 'feature' not in h5f[uid]: continue
                    feats = h5f[uid]['feature'][:]
                    if len(feats) != len(fixed_names): continue
                    row_dict = {"file_id": uid}
                    for n, v in zip(fixed_names, feats):
                        row_dict[n] = float(v)
                    row_dict["label_bot"] = row['label_bot']
                    row_dict["label_mid"] = row['label_mid']
                    row_dict["label_top"] = row['label_top']
                    export_data.append(row_dict)

        if export_data:
            df = pd.DataFrame(export_data)
            final_cols = ['file_id'] + fixed_names + ['label_bot', 'label_mid', 'label_top']
            final_cols = [c for c in final_cols if c in df.columns]
            df = df[final_cols]
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            return len(df)
        else:
            raise Exception("Export할 데이터가 없습니다.")

    def check_id_exists(self, unique_id):
        if not self.cursor: return False
        self.cursor.execute("SELECT 1 FROM noise_data WHERE id = ?", (unique_id,))
        return self.cursor.fetchone() is not None

    # [핵심 수정] 날짜 범위(Range) 검색 로직 적용
    def _build_query(self, filters, logic="AND"):
        conditions = []
        params = []
        
        # Date Range Filter: (year*10000 + month*100 + day) BETWEEN start AND end
        if filters.get('date_from') and filters.get('date_to'):
            try:
                # 2023-01-01 -> 20230101 (Int 변환)
                start_date = int(filters['date_from'].replace('-', ''))
                end_date = int(filters['date_to'].replace('-', ''))
                
                # SQLite 연산으로 YYYYMMDD 형태 정수 생성 후 비교
                conditions.append("(year * 10000 + month * 100 + day) BETWEEN ? AND ?")
                params.extend([start_date, end_date])
            except ValueError:
                pass # 날짜 형식이 잘못되었으면 무시

        if filters.get('serial'): conditions.append("serial LIKE ?"); params.append(f"%{filters['serial']}%")
        if filters.get('top'): conditions.append("label_top = ?"); params.append(filters['top'])
        if filters.get('mid'): conditions.append("label_mid = ?"); params.append(filters['mid'])
        if filters.get('bot'): conditions.append("label_bot = ?"); params.append(filters['bot'])
        
        base = f"WHERE ({f' {logic} '.join(conditions)})" if conditions else ""
        
        if filters.get('status') == 'Labeled': base += (" AND" if base else " WHERE") + " is_labeled = 1"
        elif filters.get('status') == 'Unlabeled': base += (" AND" if base else " WHERE") + " is_labeled = 0"
        return base, params

    def get_total_count(self, filters, logic="AND"):
        w, p = self._build_query(filters, logic)
        self.cursor.execute(f"SELECT COUNT(*) FROM noise_data {w}", p)
        return self.cursor.fetchone()[0]

    def fetch_list(self, page=1, per_page=20, filters=None, logic="AND"):
        w, p = self._build_query(filters, logic)
        offset = (page-1)*per_page
        p.extend([per_page, offset])
        # Mid, Bot 포함
        self.cursor.execute(f"SELECT id, filename, is_labeled, label_top, label_mid, label_bot FROM noise_data {w} ORDER BY created_at DESC LIMIT ? OFFSET ?", p)
        return self.cursor.fetchall()

    def get_ids_by_label(self, col, val, limit=50):
        self.cursor.execute(f"SELECT id FROM noise_data WHERE {col} = ? LIMIT ?", (val, limit))
        return [r[0] for r in self.cursor.fetchall()]

    def update_labels(self, uid, t, m, b):
        try:
            self.cursor.execute("UPDATE noise_data SET label_top=?, label_mid=?, label_bot=?, is_labeled=1 WHERE id=?", (t, m, b, uid))
            self.conn.commit()
            return True
        except: return False

    def get_label_stats(self):
        s = {}
        if not self.cursor: return s
        for c in ['label_top', 'label_mid', 'label_bot']:
            try:
                self.cursor.execute(f"SELECT {c}, COUNT(*) FROM noise_data WHERE {c} IS NOT NULL GROUP BY {c}")
                s[c] = dict(self.cursor.fetchall())
            except: s[c] = {}
        return s
    
    

# ==========================================
# 2. Worker Thread for Ingestion (Updated)
# ==========================================
class IngestionWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal()
    
    def __init__(self, folder_paths, db_manager):
        super().__init__()
        self.folder_paths = folder_paths
        self.db = db_manager

    def parse_feature_string_to_dict(self, feat_str):
        # "SPL=70.5&A_B1=0.2..." -> {"SPL": 70.5, "A_B1": 0.2}
        parsed = {}
        if not feat_str: return parsed
        try:
            pairs = feat_str.split('&')
            for pair in pairs:
                if '=' in pair:
                    k, v = pair.split('=', 1)
                    try: parsed[k] = float(v)
                    except: pass
        except: pass
        return parsed

    def run(self):
        self.progress.emit("Scanning folders...")
        sw_files = []
        for root_path in self.folder_paths:
            for root, dirs, files in os.walk(root_path):
                for f in files:
                    if "_SW_" in f and f.endswith(".json"):
                        sw_files.append(os.path.join(root, f))
        
        total = len(sw_files)
        self.progress.emit(f"Found {total} files. Processing (Fixed Rule Mode)...")
        count = 0
        pattern = re.compile(r"(\d{8})_(\d{6})_([A-Za-z0-9]+)")

        # [규칙 정의]
        target_channels = ['ch_1', 'ch_2', 'ch_3', 'ch_4']
        metrics = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6']

        for sw_path in sw_files:
            filename = os.path.basename(sw_path)
            try:
                match = pattern.search(filename)
                if not match: continue
                date_part, time_part, serial_part = match.groups()
                unique_id = f"{date_part}_{time_part}_{serial_part}"
                
                if self.db.check_id_exists(unique_id): continue

                # Load Signal
                with open(sw_path, 'rb') as f: sw_json = orjson.loads(f.read())
                sr = sw_json.get("SamplesPerSecond", 16000)
                ts_data = sw_json.get("ts", {})
                raw_data = {}
                for k, v in ts_data.items():
                    if k.startswith("ch_"):
                        raw_data[k] = np.array(v, dtype=np.float32)
                if not raw_data: continue

                # Load Feature File
                sf_path = sw_path.replace("_SW_", "_SF_")
                feats_by_ch = {} # {"ch_1": {"SPL": 70, "A_B1": 0.1...}, ...}
                
                if os.path.exists(sf_path):
                    with open(sf_path, 'rb') as f: sf_json = orjson.loads(f.read())
                    for k, v in sf_json.items():
                        if "Feature" in k:
                            ch_num = re.search(r"Ch(\d+)", k)
                            if ch_num:
                                ch_key = f"ch_{ch_num.group(1)}"
                                feats_by_ch[ch_key] = self.parse_feature_string_to_dict(v)

                # [핵심] 규칙대로 값 강제 추출 (Flattening)
                flat_values = []
                
                for ch in target_channels:
                    # 해당 채널의 피쳐 딕셔너리 (없으면 빈 dict)
                    ch_feats = feats_by_ch.get(ch, {})
                    
                    # 1. SPL
                    val = ch_feats.get('SPL', 0.0) # 없으면 0.0
                    flat_values.append(val)
                    
                    # 2. Metrics x Bands
                    for m in metrics:
                        for b in bands:
                            key = f"{m}_{b}" # 예: A_B1
                            val = ch_feats.get(key, 0.0)
                            flat_values.append(val)

                # Float32 Array 변환
                feat_array = np.array(flat_values, dtype=np.float32)

                # DB Insert (이름 안 넘김)
                duration = len(list(raw_data.values())[0]) / sr
                meta = {
                    'id': unique_id, 'filename': filename, 'path': sw_path,
                    'year': int(date_part[:4]), 'month': int(date_part[4:6]), 'day': int(date_part[6:8]),
                    'serial': serial_part, 'duration': duration, 'sr': sr
                }
                
                if self.db.insert_record(meta, raw_data, feat_array):
                    count += 1
                    if count % 10 == 0: self.progress.emit(f"Ingested {count}/{total}")

            except Exception as e:
                self.progress.emit(f"[Error] {filename}: {str(e)}")
        
        self.progress.emit(f"Done. {count} files added.")
        self.finished.emit()
# ==========================================
# 3. GUI Modules
# ==========================================
class DataIngesterTab(QWidget):
    def __init__(self, db_manager):
        super().__init__()
        self.db = db_manager
        self.setAcceptDrops(True) # 드래그 앤 드롭 활성화
        
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        # --- UI Components ---
        
        # 1. Control Area (Buttons)
        btn_group = QGroupBox("폴더 관리")
        btn_layout = QHBoxLayout()
        
        # [기능 1] 다중 폴더 선택 버튼
        self.btn_add = QPushButton("📂 폴더 추가 (다중 선택 가능)")
        self.btn_add.clicked.connect(self.add_folders_multi)
        
        # [기능 2] 리스트에 있는 폴더의 하위 폴더들을 찾아서 리스트에 펼치기
        self.btn_expand = QPushButton("관리 편의: 하위 폴더 펼치기")
        self.btn_expand.setToolTip("리스트에 있는 폴더 안의 바로 아래 폴더들을 찾아서 리스트에 추가합니다.")
        self.btn_expand.clicked.connect(self.expand_subfolders)
        
        self.btn_remove = QPushButton("🗑️ 선택 제거")
        self.btn_remove.clicked.connect(self.remove_selection)
        
        self.btn_clear = QPushButton("초기화")
        self.btn_clear.clicked.connect(self.clear_all)
        
        btn_layout.addWidget(self.btn_add)
        btn_layout.addWidget(self.btn_expand)
        btn_layout.addWidget(self.btn_remove)
        btn_layout.addWidget(self.btn_clear)
        btn_group.setLayout(btn_layout)
        self.layout.addWidget(btn_group)

        # 2. Target List (QListWidget)
        self.lbl_info = QLabel("대상 폴더 목록 (드래그 앤 드롭 가능 / 실제 변환 시에는 하위 폴더도 모두 포함됨):")
        self.layout.addWidget(self.lbl_info)
        
        self.list_targets = QListWidget()
        self.list_targets.setSelectionMode(QAbstractItemView.ExtendedSelection) # 다중 선택 모드
        self.layout.addWidget(self.list_targets)

        # 3. Action Area
        self.btn_run = QPushButton("🚀 데이터 변환 시작 (Start Ingestion)")
        self.btn_run.setFixedHeight(50)
        self.btn_run.setStyleSheet("font-weight: bold; font-size: 15px; background-color: #007ACC; color: white;")
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self.run_ingestion)
        self.layout.addWidget(self.btn_run)
        
        # 4. Log View
        self.log_view = QTextEdit()
        self.log_view.setPlaceholderText("로그가 여기에 표시됩니다...")
        self.log_view.setReadOnly(True)
        self.layout.addWidget(self.log_view)

    def add_folders_multi(self):
        # [핵심 수정] 다중 선택을 위해 QFileDialog 인스턴스를 직접 생성하고 옵션 조정
        dlg = QFileDialog(self)
        dlg.setFileMode(QFileDialog.Directory)
        dlg.setOption(QFileDialog.DontUseNativeDialog, True) # 윈도우 기본창 대신 Qt창 사용 (다중선택 지원)
        dlg.setWindowTitle("변환할 폴더들을 선택하세요 (Ctrl+클릭으로 다중 선택)")
        
        # 트리 뷰에서 폴더 선택 가능하도록 설정
        for view in dlg.findChildren(QAbstractItemView):
            view.setSelectionMode(QAbstractItemView.ExtendedSelection)

        if dlg.exec_():
            folders = dlg.selectedFiles()
            self.add_paths_to_list(folders)

    def add_paths_to_list(self, paths):
        # 중복 체크 후 리스트에 추가하는 공통 함수
        existing_items = set(self.list_targets.item(i).text() for i in range(self.list_targets.count()))
        
        added_count = 0
        for path in paths:
            path = os.path.normpath(path)
            if path not in existing_items and os.path.isdir(path):
                self.list_targets.addItem(path)
                added_count += 1
                
                # DB 연결 (첫 폴더 기준)
                if not self.db.conn:
                    self.db.connect_db(path)
                    self.log_view.append(f"✅ DB 초기화 위치: {path}")

        if added_count > 0:
            self.log_view.append(f"➕ {added_count}개 폴더가 추가되었습니다.")
        self.check_ready_status()

    def expand_subfolders(self):
        # 리스트에 있는 폴더들을 스캔해서 바로 아래 하위 폴더들로 교체
        count = self.list_targets.count()
        if count == 0:
            QMessageBox.information(self, "알림", "리스트가 비어있습니다.")
            return

        new_paths = []
        rows_to_remove = []

        for i in range(count):
            parent_path = self.list_targets.item(i).text()
            try:
                # 바로 아래 하위 폴더들 검색
                subdirs = [os.path.join(parent_path, d) for d in os.listdir(parent_path) if os.path.isdir(os.path.join(parent_path, d))]
                if subdirs:
                    new_paths.extend(subdirs)
                    rows_to_remove.append(i) # 하위 폴더가 있으면 부모는 리스트에서 제거 (중복 방지)
            except Exception as e:
                self.log_view.append(f"⚠️ 스캔 오류 ({parent_path}): {e}")

        if not new_paths:
            QMessageBox.information(self, "알림", "추가할 하위 폴더가 없습니다.")
            return

        # 기존 부모 항목 제거 (뒤에서부터 제거해야 인덱스 안 꼬임)
        for i in sorted(rows_to_remove, reverse=True):
            self.list_targets.takeItem(i)

        # 새 하위 폴더들 추가
        self.add_paths_to_list(new_paths)
        self.log_view.append(f"📂 하위 폴더 {len(new_paths)}개로 펼치기 완료.")

    def remove_selection(self):
        selected_items = self.list_targets.selectedItems()
        if not selected_items: return
        for item in selected_items:
            self.list_targets.takeItem(self.list_targets.row(item))
        self.check_ready_status()

    def clear_all(self):
        self.list_targets.clear()
        self.check_ready_status()

    def check_ready_status(self):
        cnt = self.list_targets.count()
        if cnt > 0:
            self.btn_run.setEnabled(True)
            self.btn_run.setText(f"🚀 데이터 변환 시작 (대상: {cnt}개 경로)")
        else:
            self.btn_run.setEnabled(False)
            self.btn_run.setText("🚀 데이터 변환 시작")

    # --- Drag & Drop ---
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        files = []
        for u in event.mimeData().urls():
            # 윈도우 경로 호환성 처리
            path = u.toLocalFile()
            if os.path.isdir(path):
                files.append(path)
        
        if files:
            self.add_paths_to_list(files)
        else:
            self.log_view.append("⚠️ 폴더만 추가할 수 있습니다.")

    def run_ingestion(self):
        targets = [self.list_targets.item(i).text() for i in range(self.list_targets.count())]
        self.btn_run.setEnabled(False)
        self.log_view.append("--- 변환 작업 시작 ---")
        
        # Worker에 경로 리스트 전달
        self.worker = IngestionWorker(targets, self.db)
        self.worker.progress.connect(self.log_view.append)
        self.worker.finished.connect(lambda: self.check_ready_status())
        self.worker.start()

class DataLabelerTab(QWidget):
    def __init__(self, db_manager):
        super().__init__()
        self.db = db_manager
        
        self.current_page = 1
        self.items_per_page = 20
        self.total_items = 0
        self.total_pages = 1
        
        self.current_id = None
        self.current_sr = 51200
        self.current_data = {}
        self.current_feat_arr = []
        self.current_feat_names = []
        
        self.is_loading_table = False 
        
        self.top_labels = ["Bearing", "Others"]
        self.mid_labels = ["Normal", "Noise", "Vibration"]
        self.bot_labels = ["OK", "NG", "Unknown"]
        
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)
        
        self.setStyleSheet("""
            QListWidget::item:selected {
                background-color: #0078D7;
                color: white;
                border: 1px solid #005A9E;
            }
            QListWidget::item:hover { background-color: #333333; }
        """)
        
        # --- Zone 1 ---
        self.zone1 = QGroupBox("Zone 1: DB & Search")
        z1_layout = QVBoxLayout()
        
        db_layout = QHBoxLayout()
        self.btn_open_db = QPushButton("📂 Open DB")
        self.btn_open_db.clicked.connect(self.open_db_file)
        self.lbl_db_name = QLabel("No DB Loaded")
        self.lbl_db_name.setStyleSheet("color: #aaa; font-size: 11px;")
        db_layout.addWidget(self.btn_open_db); db_layout.addWidget(self.lbl_db_name)
        z1_layout.addLayout(db_layout)
        
        search_grp = QGroupBox("Search Filters")
        search_grid = QGridLayout()
        self.chk_date_enable = QCheckBox("Date Filter:")
        search_grid.addWidget(self.chk_date_enable, 0, 0)
        date_layout = QHBoxLayout()
        self.date_from = QDateEdit(); self.date_from.setCalendarPopup(True); self.date_from.setDisplayFormat("yyyy-MM-dd"); self.date_from.setDate(QDate.currentDate().addMonths(-1))
        self.date_to = QDateEdit(); self.date_to.setCalendarPopup(True); self.date_to.setDisplayFormat("yyyy-MM-dd"); self.date_to.setDate(QDate.currentDate())
        date_layout.addWidget(QLabel("From")); date_layout.addWidget(self.date_from); date_layout.addWidget(QLabel("To")); date_layout.addWidget(self.date_to)
        search_grid.addLayout(date_layout, 0, 1)
        search_grid.addWidget(QLabel("Serial:"), 1, 0)
        self.txt_serial = QLineEdit(); self.txt_serial.setPlaceholderText("ex) R1K6...")
        search_grid.addWidget(self.txt_serial, 1, 1)
        search_grid.addWidget(QLabel("Top:"), 2, 0); self.combo_filter_top = QComboBox(); search_grid.addWidget(self.combo_filter_top, 2, 1)
        search_grid.addWidget(QLabel("Mid:"), 3, 0); self.combo_filter_mid = QComboBox(); search_grid.addWidget(self.combo_filter_mid, 3, 1)
        search_grid.addWidget(QLabel("Bot:"), 4, 0); self.combo_filter_bot = QComboBox(); search_grid.addWidget(self.combo_filter_bot, 4, 1)
        search_grid.addWidget(QLabel("Status:"), 5, 0); self.combo_status = QComboBox()
        self.combo_status.addItems(["All", "Unlabeled", "Labeled"])
        search_grid.addWidget(self.combo_status, 5, 1)
        logic_layout = QHBoxLayout()
        self.rb_and = QRadioButton("AND"); self.rb_or = QRadioButton("OR"); self.rb_and.setChecked(True)
        logic_layout.addWidget(self.rb_and); logic_layout.addWidget(self.rb_or)
        search_grid.addLayout(logic_layout, 6, 0, 1, 2)
        self.btn_search = QPushButton("🔍 Apply Filter"); self.btn_search.clicked.connect(self.reset_and_load)
        search_grid.addWidget(self.btn_search, 7, 0, 1, 2)
        search_grp.setLayout(search_grid); z1_layout.addWidget(search_grp)

        self.table = QTableWidget(); self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["ID", "Top", "Mid", "Bot", "Status"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.verticalHeader().setVisible(False)
        self.table.cellClicked.connect(self.on_table_click)
        self.table.currentCellChanged.connect(self.on_current_cell_changed)
        self.table.itemChanged.connect(self.on_table_item_changed)
        self.table.installEventFilter(self)
        
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        z1_layout.addWidget(self.table)
        
        page_layout = QHBoxLayout()
        self.btn_prev = QPushButton("◀"); self.btn_prev.clicked.connect(self.go_prev_page)
        self.lbl_page = QLabel("0 / 0"); self.lbl_page.setAlignment(Qt.AlignCenter); self.lbl_page.setStyleSheet("font-weight: bold; color: #00FF00;")
        self.btn_next = QPushButton("▶"); self.btn_next.clicked.connect(self.go_next_page)
        page_layout.addWidget(self.btn_prev); page_layout.addWidget(self.lbl_page); page_layout.addWidget(self.btn_next)
        z1_layout.addLayout(page_layout); self.zone1.setLayout(z1_layout)

        # --- Zone 2: Visualization ---
        self.zone2 = QGroupBox("Zone 2: Visualization")
        z2_layout = QVBoxLayout()
        chk_style = "QCheckBox { color: black; background-color: #dddddd; padding: 4px; border-radius: 4px; font-weight: bold; }"
        
        ctrl_layout = QHBoxLayout()
        self.chk_ch1 = QCheckBox("Ch1"); self.chk_ch1.setChecked(True); self.chk_ch1.setStyleSheet(chk_style)
        self.chk_ch2 = QCheckBox("Ch2"); self.chk_ch2.setChecked(True); self.chk_ch2.setStyleSheet(chk_style)
        self.chk_ch3 = QCheckBox("Ch3"); self.chk_ch3.setChecked(True); self.chk_ch3.setStyleSheet(chk_style)
        self.chk_ch4 = QCheckBox("Ch4"); self.chk_ch4.setStyleSheet(chk_style)
        self.chk_ch1.stateChanged.connect(self.update_plots); self.chk_ch2.stateChanged.connect(self.update_plots)
        self.chk_ch3.stateChanged.connect(self.update_plots); self.chk_ch4.stateChanged.connect(self.update_plots)
        ctrl_layout.addWidget(self.chk_ch1); ctrl_layout.addWidget(self.chk_ch2)
        ctrl_layout.addWidget(self.chk_ch3); ctrl_layout.addWidget(self.chk_ch4); z2_layout.addLayout(ctrl_layout)
        
        ctrl_layout2 = QHBoxLayout()
        self.chk_sync = QCheckBox("Sync X-Axis"); self.chk_sync.setStyleSheet(chk_style)
        self.chk_sync.stateChanged.connect(self.update_plots)
        
        lbl_cmap = QLabel("Color:")
        self.combo_cmap = QComboBox()
        self.combo_cmap.addItems(['inferno', 'viridis', 'plasma', 'magma', 'jet', 'hot', 'cool', 'spring', 'summer', 'autumn', 'winter'])
        self.combo_cmap.currentTextChanged.connect(self.update_plots)
        
        self.btn_feat = QPushButton("View Feature"); self.btn_feat.clicked.connect(self.show_feature_popup)
        
        ctrl_layout2.addWidget(self.chk_sync); ctrl_layout2.addSpacing(20); ctrl_layout2.addWidget(lbl_cmap); ctrl_layout2.addWidget(self.combo_cmap)
        ctrl_layout2.addStretch(1); ctrl_layout2.addWidget(self.btn_feat)
        z2_layout.addLayout(ctrl_layout2)
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.plot_container = QWidget()
        self.plot_layout = QVBoxLayout()
        self.plot_layout.setSpacing(20)
        self.plot_container.setLayout(self.plot_layout)
        self.scroll_area.setWidget(self.plot_container)
        z2_layout.addWidget(self.scroll_area)
        self.zone2.setLayout(z2_layout)

        # --- Zone 3,4,5: Labels ---
        self.zone_labels = QWidget()
        label_layout = QHBoxLayout()
        self.zone_labels.setLayout(label_layout)
        
        def create_label_zone(title, label_col, category_key):
            grp = QGroupBox(title)
            v = QVBoxLayout()
            btn_box = QHBoxLayout()
            btn_add = QPushButton("➕ Add"); btn_del = QPushButton("➖ Del")
            btn_box.addWidget(btn_add); btn_box.addWidget(btn_del)
            v.addLayout(btn_box)
            lst = QListWidget()
            lst.setDragDropMode(QAbstractItemView.InternalMove)
            lst.setFixedHeight(120) 
            lst.model().rowsMoved.connect(lambda: self.save_label_order(category_key, lst))
            btn_add.clicked.connect(lambda: self.add_new_label(lst, category_key))
            btn_del.clicked.connect(lambda: self.delete_label(lst, label_col, category_key))
            lbl_prev = QLabel("▼ Comparison:")
            lbl_prev.setStyleSheet("color: #888; font-size: 10px;")
            lst_prev = QListWidget()
            lst_prev.setStyleSheet("font-size: 11px; color: #ddd; background-color: #222;")
            lst.itemClicked.connect(lambda item: self.load_preview(label_col, item.text(), lst_prev))
            lst_prev.itemClicked.connect(lambda item: self.load_data(item.text()))
            v.addWidget(lst); v.addWidget(lbl_prev); v.addWidget(lst_prev)
            grp.setLayout(v)
            return grp, lst, lst_prev

        self.z3_grp, self.list_bot, self.prev_bot = create_label_zone("Zone 3: Bottom", "label_bot", "bot")
        self.z4_grp, self.list_mid, self.prev_mid = create_label_zone("Zone 4: Mid", "label_mid", "mid")
        self.z5_grp, self.list_top, self.prev_top = create_label_zone("Zone 5: Top", "label_top", "top")
        
        self.btn_save = QPushButton("💾 Set Label (Save & Next)")
        self.btn_save.setStyleSheet("background-color: #2E7D32; color: white; font-weight: bold; padding: 10px;")
        self.btn_save.clicked.connect(self.save_and_next)
        self.btn_export = QPushButton("📤 Export Data")
        self.btn_export.clicked.connect(self.export_data)
        self.z5_grp.layout().addWidget(self.btn_save)
        self.z5_grp.layout().addWidget(self.btn_export)
        label_layout.addWidget(self.z3_grp); label_layout.addWidget(self.z4_grp); label_layout.addWidget(self.z5_grp)
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.zone1); splitter.addWidget(self.zone2); splitter.addWidget(self.zone_labels)
        splitter.setSizes([350, 600, 350])
        main_layout.addWidget(splitter)

    # --- Event Filters ---
    def eventFilter(self, source, event):
        if source == self.table and event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_Down:
                current_row = self.table.currentRow()
                if current_row == self.table.rowCount() - 1 and self.current_page < self.total_pages:
                    self.go_next_page()
                    self.table.selectRow(0); self.on_table_click(0, 0)
                    return True
            elif event.key() == Qt.Key_Up:
                current_row = self.table.currentRow()
                if current_row == 0 and self.current_page > 1:
                    self.go_prev_page()
                    last_row = self.table.rowCount() - 1
                    self.table.selectRow(last_row); self.on_table_click(last_row, 0)
                    return True
        return super().eventFilter(source, event)

    # --- Handlers ---
    def on_current_cell_changed(self, current_row, current_col, previous_row, previous_col):
        if self.is_loading_table or current_row < 0: return
        item = self.table.item(current_row, 0)
        if item: self.load_data(item.text())

    def on_table_click(self, row, col):
        item = self.table.item(row, 0)
        if item: self.load_data(item.text())

    def on_table_item_changed(self, item):
        if self.is_loading_table or item.column() in [0, 4]: return
        uid = self.table.item(item.row(), 0).text()
        new_val = item.text().strip()
        col_map = {1: ('label_top', 'top'), 2: ('label_mid', 'mid'), 3: ('label_bot', 'bot')}
        if item.column() not in col_map: return
        db_col, category = col_map[item.column()]
        if self.db.update_single_label(uid, db_col, new_val):
            if new_val and self.db.ensure_label_exists(category, new_val): self.load_labels_from_db()
            top_item = self.table.item(item.row(), 1)
            has_top = top_item and top_item.text().strip() != ""
            status_item = self.table.item(item.row(), 4)
            if has_top: status_item.setText("Labeled"); status_item.setForeground(QColor("#00FF00"))
            else: status_item.setText("Unlabeled"); status_item.setForeground(QColor("#FFaa00"))
            self.refresh_stats()

    # --- Main Logic ---
    def open_db_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Select DB File", "", "SQLite DB (*.db)")
        if fname:
            self.db.connect_db(fname); self.lbl_db_name.setText(os.path.basename(fname))
            self.db.sync_missing_labels()
            self.load_labels_from_db()
            self.reset_and_load(); self.refresh_stats()

    def load_labels_from_db(self):
        def fill_list(qlist, category, defaults):
            qlist.clear()
            saved = self.db.fetch_label_settings(category)
            if not saved: saved = defaults; self.db.save_label_settings(category, saved)
            qlist.addItems(saved)
        fill_list(self.list_top, 'top', self.top_labels)
        fill_list(self.list_mid, 'mid', self.mid_labels)
        fill_list(self.list_bot, 'bot', self.bot_labels)
        self.update_combos()

    def update_combos(self):
        def fill_combo(combo, qlist):
            combo.clear(); combo.addItem("All")
            items = [qlist.item(i).text().split(' (')[0] for i in range(qlist.count())]
            combo.addItems(items)
        fill_combo(self.combo_filter_top, self.list_top)
        fill_combo(self.combo_filter_mid, self.list_mid)
        fill_combo(self.combo_filter_bot, self.list_bot)

    def save_label_order(self, category, list_widget):
        items = [list_widget.item(i).text().split(' (')[0] for i in range(list_widget.count())]
        self.db.save_label_settings(category, items)

    def add_new_label(self, list_widget, category):
        text, ok = QInputDialog.getText(self, "Add Label", "New Label Name:")
        if ok and text:
            items = [list_widget.item(x).text().split(' (')[0] for x in range(list_widget.count())]
            if text in items: QMessageBox.warning(self, "Duplicate", "Label already exists."); return
            list_widget.addItem(text); self.save_label_order(category, list_widget); self.update_combos()

    def delete_label(self, list_widget, label_col, category):
        item = list_widget.currentItem()
        if not item: QMessageBox.warning(self, "Select", "Please select a label to delete."); return
        label_text = item.text().split(' (')[0]
        reply = QMessageBox.question(self, "Delete Label", f"Delete '{label_text}'?\nRelated data will become Unlabeled.", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            if self.db.remove_label_from_records(label_col, label_text):
                list_widget.takeItem(list_widget.row(item)); self.save_label_order(category, list_widget)
                self.refresh_stats(); self.update_combos(); self.load_list()
                QMessageBox.information(self, "Done", "Label deleted and list updated.")
            else: QMessageBox.critical(self, "Error", "DB Update Failed.")

    def load_preview(self, label_col, label_val, preview_widget):
        label_text = label_val.split(' (')[0]
        ids = self.db.get_ids_by_label(label_col, label_text)
        preview_widget.clear()
        if not ids: preview_widget.addItem("No data found.")
        else:
            for uid in ids: preview_widget.addItem(uid)

    def get_filter_opts(self):
        opts = {}
        if self.chk_date_enable.isChecked():
            opts['date_from'] = self.date_from.date().toString("yyyy-MM-dd")
            opts['date_to'] = self.date_to.date().toString("yyyy-MM-dd")
        if self.txt_serial.text().strip(): opts['serial'] = self.txt_serial.text().strip()
        if self.combo_filter_top.currentText() != "All": opts['top'] = self.combo_filter_top.currentText()
        if self.combo_filter_mid.currentText() != "All": opts['mid'] = self.combo_filter_mid.currentText()
        if self.combo_filter_bot.currentText() != "All": opts['bot'] = self.combo_filter_bot.currentText()
        opts['status'] = self.combo_status.currentText()
        return opts

    def get_logic_operator(self): return "AND" if self.rb_and.isChecked() else "OR"
    def reset_and_load(self): self.current_page = 1; self.load_list()

    def load_list(self):
        if not self.db.conn: return
        self.is_loading_table = True
        try:
            opts = self.get_filter_opts(); logic = self.get_logic_operator()
            total = self.db.get_total_count(opts, logic)
            self.total_items = total
            self.total_pages = max(1, (total + self.items_per_page - 1) // self.items_per_page)
            rows = self.db.fetch_list(self.current_page, self.items_per_page, opts, logic)
            self.table.setRowCount(len(rows))
            for r, row in enumerate(rows):
                self.table.setItem(r, 0, QTableWidgetItem(str(row['id'])))
                self.table.setItem(r, 1, QTableWidgetItem(row['label_top'] if row['label_top'] else ""))
                self.table.setItem(r, 2, QTableWidgetItem(row['label_mid'] if row['label_mid'] else ""))
                self.table.setItem(r, 3, QTableWidgetItem(row['label_bot'] if row['label_bot'] else ""))
                status = "Labeled" if row['is_labeled'] else "Unlabeled"
                item = QTableWidgetItem(status); item.setForeground(QColor("#00FF00") if row['is_labeled'] else QColor("#FFaa00"))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(r, 4, item)
            self.lbl_page.setText(f"Page {self.current_page} / {self.total_pages} (Total {self.total_items})")
            self.btn_prev.setEnabled(self.current_page > 1); self.btn_next.setEnabled(self.current_page < self.total_pages)
        finally: self.is_loading_table = False

    def go_prev_page(self): 
        if self.current_page > 1: self.current_page -= 1; self.load_list()
    def go_next_page(self): 
        if self.current_page < self.total_pages: self.current_page += 1; self.load_list()
    
    def load_data(self, uid):
        self.current_id = uid
        raws, feat_arr, feat_names = self.db.get_signal_data(uid)
        if raws is None: return
        self.current_data = raws; self.current_feat_arr = feat_arr; self.current_feat_names = feat_names
        self.update_plots()

    def update_plots(self):
        while self.plot_layout.count():
            child = self.plot_layout.takeAt(0)
            if child.widget(): child.widget().deleteLater()
        
        channels = []
        if self.chk_ch1.isChecked(): channels.append(('ch_1', 'Ch1'))
        if self.chk_ch2.isChecked(): channels.append(('ch_2', 'Ch2'))
        if self.chk_ch3.isChecked(): channels.append(('ch_3', 'Ch3'))
        if self.chk_ch4.isChecked(): channels.append(('ch_4', 'Ch4'))
        
        prev_plot = None
        cmap_name = self.combo_cmap.currentText()
        try: colormap = pg.colormap.get(cmap_name)
        except: colormap = pg.colormap.get('inferno')
        is_sync = self.chk_sync.isChecked()
        
        for idx, (key, title) in enumerate(channels):
            if key not in self.current_data: continue
            sig = self.current_data[key]
            
            container = QWidget()
            cont_layout = QVBoxLayout()
            cont_layout.setContentsMargins(0,0,0,0); cont_layout.setSpacing(2)
            container.setLayout(cont_layout)
            
            title_row = QWidget(); title_layout = QHBoxLayout(); title_layout.setContentsMargins(5,0,5,0); title_row.setLayout(title_layout)
            lbl_title = QLabel(f"<b>{title}</b>")
            
            # [재생]
            btn_play = QPushButton("▶ Play")
            btn_play.setFixedSize(60, 24)
            btn_play.setStyleSheet("background-color: #4CAF50; color: white; border: none; border-radius: 3px;")
            btn_play.clicked.connect(lambda checked, s=sig: self.play_audio(s))
            
            # [정지]
            btn_stop = QPushButton("■ Stop")
            btn_stop.setFixedSize(60, 24)
            btn_stop.setStyleSheet("background-color: #E53935; color: white; border: none; border-radius: 3px;")
            btn_stop.clicked.connect(self.stop_audio)
            
            title_layout.addWidget(lbl_title); title_layout.addWidget(btn_play); title_layout.addWidget(btn_stop); title_layout.addStretch()
            cont_layout.addWidget(title_row)
            
            p_widget = pg.PlotWidget(); p_widget.setMinimumHeight(200); p = p_widget.getPlotItem()
            f, t, Zxx = signal.stft(sig, fs=self.current_sr, nperseg=512, noverlap=256)
            spec_db = 20 * np.log10(np.abs(Zxx) + 1e-6)
            p.setLabel('left', 'Freq', units='Hz'); p.setLabel('bottom', 'Time', units='s')
            img = pg.ImageItem(); p.addItem(img); img.setImage(spec_db.T)
            img.setRect(QRectF(t[0], f[0], t[-1]-t[0], f[-1]-f[0]))
            img.setLookupTable(colormap.getLookupTable())
            img.setLevels([np.max(spec_db)-80, np.max(spec_db)])
            cont_layout.addWidget(p_widget)
            
            if is_sync and prev_plot: p.setXLink(prev_plot)
            self.plot_layout.addWidget(container)
            prev_plot = p

    def play_audio(self, data):
        try: sd.stop(); sd.play(data, self.current_sr)
        except Exception as e: print(f"Audio Error: {e}")

    def stop_audio(self):
        try: sd.stop()
        except Exception as e: print(f"Audio Error: {e}")

    def show_feature_popup(self):
        if self.current_feat_arr is None or len(self.current_feat_arr) == 0: return
        msg = QDialog(self); msg.setWindowTitle("Features"); msg.resize(400, 500)
        txt = QTextEdit(); txt.setReadOnly(True)
        content = "Index | Name : Value\n" + "-"*35 + "\n"
        for i, val in enumerate(self.current_feat_arr):
            name = self.current_feat_names[i] if i < len(self.current_feat_names) else f"Feat_{i}"
            content += f"[{i:03d}] {name} : {val:.5f}\n"
        txt.setText(content); l = QVBoxLayout(); l.addWidget(txt); msg.setLayout(l); msg.exec_()

    def refresh_stats(self):
        if not self.db.conn: return
        stats = self.db.get_label_stats()
        def up(qlist, s_dict):
            for i in range(qlist.count()):
                it = qlist.item(i); base = it.text().split(' (')[0]
                it.setText(f"{base} ({s_dict.get(base, 0)})")
        up(self.list_top, stats.get('label_top', {}))
        up(self.list_mid, stats.get('label_mid', {}))
        up(self.list_bot, stats.get('label_bot', {}))

    def save_and_next(self):
        if not self.current_id: return
        top = self.list_top.currentItem()
        mid = self.list_mid.currentItem()
        bot = self.list_bot.currentItem()
        if not top: QMessageBox.warning(self, "Warning", "Top Label is mandatory!"); return
        
        t = top.text().split(' (')[0]
        m = mid.text().split(' (')[0] if mid else "Unlabeled"
        b = bot.text().split(' (')[0] if bot else "Unlabeled"
        
        if self.db.update_labels(self.current_id, t, m, b):
            self.refresh_stats()
            self.is_loading_table = True
            curr_row = self.table.currentRow()
            if curr_row >= 0:
                self.table.item(curr_row, 1).setText(t); self.table.item(curr_row, 2).setText(m); self.table.item(curr_row, 3).setText(b)
                status_item = self.table.item(curr_row, 4); status_item.setText("Labeled"); status_item.setForeground(QColor("#00FF00"))
            self.is_loading_table = False
            
            if curr_row < self.table.rowCount() - 1:
                self.table.selectRow(curr_row + 1); self.on_table_click(curr_row + 1, 0)
            else:
                if self.current_page < self.total_pages:
                    self.go_next_page(); self.table.selectRow(0); self.on_table_click(0, 0)
                else:
                    QMessageBox.information(self, "Done", "Last item reached.")
        else: QMessageBox.critical(self, "Error", "Failed to save label.")

    def export_data(self):
        fname, _ = QFileDialog.getSaveFileName(self, "Save Export File", "", "CSV Files (*.csv)")
        if fname:
            try:
                count = self.db.export_training_data(fname)
                QMessageBox.information(self, "Success", f"Exported {count} items.\n{fname}")
            except Exception as e: QMessageBox.critical(self, "Error", str(e))
            
            
            

# ==========================================
# 4. Main Window (Style Fix)
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Factory Noise Labeling Platform")
        self.resize(1400, 900)
        
        self.db_manager = DataManager()
        
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        self.tab_ingester = DataIngesterTab(self.db_manager)
        self.tab_labeler = DataLabelerTab(self.db_manager)
        
        self.tabs.addTab(self.tab_ingester, "Data Ingester")
        self.tabs.addTab(self.tab_labeler, "Data Labeler")
        
        # [수정됨] QMessageBox 및 전체 스타일시트 보강
        # QWidget 전역 설정을 통해 기본 글자색을 잡고,
        # QMessageBox 컴포넌트들의 색상을 명시적으로 지정하여 가독성 확보
        self.setStyleSheet("""
            /* 전체 앱 기본 스타일 */
            QMainWindow, QWidget { 
                background-color: #2b2b2b; 
                color: #ffffff; 
                font-size: 14px;
            }
            
            /* 탭 위젯 스타일 */
            QTabWidget::pane { border: 1px solid #444; }
            QTabBar::tab { background: #333; color: #aaa; padding: 8px 20px; }
            QTabBar::tab:selected { background: #555; color: #fff; font-weight: bold; }
            
            /* 그룹박스 스타일 */
            QGroupBox { 
                font-weight: bold; 
                border: 1px solid #555; 
                margin-top: 10px; 
                padding-top: 10px; 
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }
            
            /* 테이블 위젯 스타일 */
            QTableWidget { 
                background-color: #1e1e1e; 
                color: #fff; 
                gridline-color: #444; 
                selection-background-color: #0078D7;
            }
            QHeaderView::section { background-color: #444; color: #fff; padding: 4px; }
            QTableWidget QTableCornerButton::section { background-color: #444; }
            
            /* 리스트 위젯 스타일 */
            QListWidget { background-color: #1e1e1e; color: #fff; border: 1px solid #555; }
            
            /* 버튼 스타일 */
            QPushButton { 
                background-color: #555; 
                color: #fff; 
                border: 1px solid #777; 
                border-radius: 3px; 
                padding: 6px; 
            }
            QPushButton:hover { background-color: #666; border-color: #999; }
            QPushButton:pressed { background-color: #444; }
            QPushButton:disabled { background-color: #333; color: #777; }
            
            /* 텍스트 에디트/라인 에디트 */
            QTextEdit, QLineEdit { 
                background-color: #1e1e1e; 
                color: #fff; 
                border: 1px solid #555; 
            }
            
            /* 콤보박스 */
            QComboBox { 
                background-color: #1e1e1e; 
                color: white; 
                border: 1px solid #555; 
                padding: 4px; 
            }
            QComboBox QAbstractItemView {
                background-color: #1e1e1e;
                color: white;
                selection-background-color: #0078D7;
            }

            /* [핵심 수정] QMessageBox 스타일 강제 적용 */
            QMessageBox {
                background-color: #2b2b2b;
            }
            QMessageBox QLabel {
                color: #ffffff; /* 메시지 텍스트 흰색 */
                background-color: transparent;
            }
            QMessageBox QPushButton {
                min-width: 60px; /* 버튼 크기 확보 */
            }
        """)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())