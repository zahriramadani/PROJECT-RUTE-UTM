import streamlit as st
import math
import heapq
from collections import deque
import matplotlib.pyplot as plt

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="UTM SmartNav - A* Pathfinding", layout="wide")

# --- 1. DATASET 25 GEDUNG UTM (SESUAI PETA SKETSA & TABEL) ---
# Orientasi koordinat mengikuti gambar peta pengguna:
# Atas = Utara/Barat (Gerbang 0 di kanan atas, Klinik 23 di kiri atas)
NODES = {
    0:  {"name": "0. Gerbang Utama", "pos": (660, 880)},
    1:  {"name": "1. Pos Keamanan", "pos": (660, 810)},
    2:  {"name": "2. Green House", "pos": (645, 560)},
    3:  {"name": "3. Lab. Eksak (Lab. IPA Terpadu)", "pos": (685, 415)},
    4:  {"name": "4. Asrama Mahasiswa", "pos": (740, 220)},
    5:  {"name": "5. Kantin", "pos": (685, 135)},
    6:  {"name": "6. Lab. Sosial (Lab. IPS)", "pos": (765, 330)},
    7:  {"name": "7. Fakultas Ilmu Pendidikan", "pos": (475, 265)},
    8:  {"name": "8. Fakultas Teknik", "pos": (575, 145)},
    9:  {"name": "9. Fakultas ISIB (RKB E)", "pos": (325, 275)},
    10: {"name": "10. Gedung Graha Utama / Rektorat", "pos": (215, 345)},
    11: {"name": "11. Taman Kampus (Tanian)", "pos": (385, 345)},
    12: {"name": "12. Gedung Cakra", "pos": (575, 345)},
    13: {"name": "13. Fakultas Keislaman", "pos": (385, 420)},
    14: {"name": "14. Fakultas Pertanian", "pos": (260, 445)},
    15: {"name": "15. Fakultas Hukum", "pos": (475, 410)},
    16: {"name": "16. Fakultas Ekonomi dan Bisnis", "pos": (565, 465)},
    17: {"name": "17. Fakultas Kedokteran", "pos": (325, 815)},
    18: {"name": "18. Lab. Manufaktur", "pos": (105, 470)},
    19: {"name": "19. Sekretariat UKM", "pos": (445, 815)},
    20: {"name": "20. Masjid Abdur Rohman Wahid", "pos": (655, 680)},
    21: {"name": "21. Lab. Mesin", "pos": (110, 610)},
    22: {"name": "22. GOR SAS", "pos": (115, 715)},
    23: {"name": "23. Unit Pelayanan Kesehatan", "pos": (120, 825)},
    24: {"name": "24. Gedung Pertemuan", "pos": (555, 745)},
}

# Format: (asal, tujuan, jarak_meter, tipe_jalur, dua_arah)
# tipe_jalur:
#   "main"         = Garis Merah (Jalan Utama Kampus)
#   "taman"        = Garis Ungu (Taman Kampus / Tanian - khusus Pejalan Kaki & Sepeda)
#   "shortcut_bad" = Garis Hijau '>' (Jalur Pintas Kecil / Jalan Jelek)
EDGES = [
    # Jalur Masuk & Keluar Utama (Garis Merah Kembar)
    (0, 1, 45, "main", True),
    (1, 20, 110, "main", True),
    (20, 2, 95, "main", True),
    (2, 12, 135, "main", True),
    (16, 1, 220, "main", True),

    # Jalur Lingkar Atas & Kiri (24 - 19 - 17 - 23 - 22 - 21 - 18 - 10)
    (24, 1, 145, "main", True),
    (24, 19, 80, "main", True),
    (19, 17, 65, "main", True),
    (17, 23, 130, "main", True),
    (23, 22, 85, "main", True),
    (22, 21, 85, "main", True),
    (21, 18, 95, "main", True),
    (18, 10, 165, "main", True),

    # Klaster Fakultas Tengah & Bundaran Cakra (Garis Merah)
    (10, 9, 105, "main", True),
    (9, 7, 110, "main", True),
    (10, 14, 95, "main", True),
    (14, 13, 75, "main", True),
    (13, 15, 65, "main", True),
    (15, 16, 75, "main", True),
    (12, 3, 95, "main", True),
    (12, 7, 90, "main", True),

    # Lingkar Timur & Selatan: Lab Eksak - Lab Sosial - Asrama - Kantin - FT - FIP
    (3, 6, 95, "main", True),
    (6, 4, 90, "main", True),
    (4, 5, 80, "main", True),
    (5, 8, 85, "main", True),
    (8, 7, 75, "main", True),

    # Garis Ungu: Taman Kampus / Tanian (Titik 11 terhubung ke 7, 9, 10, 12, 13, 14, 15)
    (11, 7, 85, "taman", True),
    (11, 9, 80, "taman", True),
    (11, 10, 90, "taman", True),
    (11, 12, 95, "taman", True),
    (11, 13, 75, "taman", True),
    (11, 14, 85, "taman", True),
    (11, 15, 80, "taman", True),

    # Garis Hijau ('>'): Jalur Pintas Kecil & Jalan Jelek (20 -> 24 dan 12 -> 15)
    (20, 24, 45, "shortcut_bad", True),
    (12, 15, 55, "shortcut_bad", True),
]

SPEEDS = {
    "Pejalan Kaki (4.5 km/jam)": {"speed": 75.0, "bad_road_factor": 0.65, "is_vehicle": False},
    "Sepeda (15 km/jam)": {"speed": 250.0, "bad_road_factor": 0.50, "is_vehicle": False},
    "Sepeda Motor (25 km/jam)": {"speed": 416.7, "bad_road_factor": 0.35, "is_vehicle": True},
    "Mobil (20 km/jam)": {"speed": 333.3, "bad_road_factor": 0.0, "is_vehicle": True},
}

def build_graph(mode_info, allow_bad_shortcut):
    graph = {node_id: [] for node_id in NODES}
    for u, v, dist, road_type, bidir in EDGES:
        # Taman kampus (ungu) tidak bisa dilewati kendaraan bermotor
        if road_type == "taman" and mode_info["is_vehicle"]:
            continue
        # Jalur hijau ('>') tidak bisa dilewati mobil, atau jika user mematikan opsi jalur jelek
        if road_type == "shortcut_bad":
            if not allow_bad_shortcut or mode_info["bad_road_factor"] == 0.0:
                continue

        graph[u].append((v, dist, road_type))
        if bidir:
            graph[v].append((u, dist, road_type))
    return graph

# --- 2. ALGORITMA A* SEARCH & PEMBANDING BFS ---
def heuristic(id_a, id_b):
    x1, y1 = NODES[id_a]["pos"]
    x2, y2 = NODES[id_b]["pos"]
    return math.hypot(x2 - x1, y2 - y1) * 0.85  # Dijamin admissible (<= jarak jalan aktual)

def a_star_search(graph, start, goal):
    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), 0, start, [start], []))
    visited = {}
    trace_log = []

    while open_set:
        f_curr, g_curr, current, path, road_types = heapq.heappop(open_set)

        if current in visited and visited[current] <= g_curr:
            continue
        visited[current] = g_curr

        trace_log.append({
            "Urutan": len(trace_log) + 1,
            "Simpul Dievaluasi": NODES[current]["name"],
            "g(n) Aktual (m)": round(g_curr, 1),
            "h(n) Heuristik (m)": round(heuristic(current, goal), 1),
            "f(n) = g + h (m)": round(f_curr, 1),
        })

        if current == goal:
            return path, g_curr, road_types, trace_log

        for neighbor, weight, r_type in graph[current]:
            g_next = g_curr + weight
            if neighbor not in visited or g_next < visited[neighbor]:
                h_next = heuristic(neighbor, goal)
                f_next = g_next + h_next
                heapq.heappush(
                    open_set,
                    (f_next, g_next, neighbor, path + [neighbor], road_types + [(current, neighbor, weight, r_type)])
                )

    return None, float("inf"), [], trace_log

def bfs_comparison(graph, start, goal):
    queue = deque([(start, [start], 0)])
    visited = {start}
    nodes_expanded = 0

    while queue:
        current, path, dist = queue.popleft()
        nodes_expanded += 1
        if current == goal:
            return path, dist, nodes_expanded
        for neighbor, weight, _ in graph[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor], dist + weight))
    return None, float("inf"), nodes_expanded

# --- 3. VISUALISASI PETA SESUAI WARNA SKETSA PENGGUNA ---
def draw_campus_map(shortest_path):
    fig, ax = plt.subplots(figsize=(11, 9))

    for u, v, dist, r_type, _ in EDGES:
        x_vals = [NODES[u]["pos"][0], NODES[v]["pos"][0]]
        y_vals = [NODES[u]["pos"][1], NODES[v]["pos"][1]]

        if r_type == "main":
            color, lw, style = "#fca5a5", 2.5, "-"       # Merah muda (Jalan Utama)
        elif r_type == "taman":
            color, lw, style = "#c084fc", 2.2, "--"      # Ungu (Taman Kampus / Tanian)
        else:
            color, lw, style = "#22c55e", 3.0, ":"       # Hijau (Jalur Pintas Kecil / Jelek '>')

        ax.plot(x_vals, y_vals, color=color, linewidth=lw, linestyle=style, zorder=1)
        mid_x, mid_y = (x_vals[0] + x_vals[1]) / 2, (y_vals[0] + y_vals[1]) / 2
        ax.text(mid_x, mid_y, f"{dist}m", fontsize=6.5, color="#475569", ha="center")

    if shortest_path and len(shortest_path) > 1:
        for i in range(len(shortest_path) - 1):
            u, v = shortest_path[i], shortest_path[i + 1]
            ax.plot(
                [NODES[u]["pos"][0], NODES[v]["pos"][0]],
                [NODES[u]["pos"][1], NODES[v]["pos"][1]],
                color="#dc2626",
                linewidth=4.5,
                zorder=3,
                label="Rute Terpendek (A*)" if i == 0 else "",
            )

    for node_id, data in NODES.items():
        x, y = data["pos"]
        if shortest_path and node_id == shortest_path[0]:
            color, size = "#16a34a", 150
        elif shortest_path and node_id == shortest_path[-1]:
            color, size = "#dc2626", 150
        elif shortest_path and node_id in shortest_path:
            color, size = "#f59e0b", 110
        elif node_id == 11:
            color, size = "#9333ea", 130
        else:
            color, size = "#1e293b", 75

        ax.scatter(x, y, color=color, s=size, zorder=4, edgecolors="white", linewidth=1.2)
        short_name = data["name"].split(" (")[0]
        ax.text(x, y + 14, short_name, fontsize=7.2, fontweight="bold", ha="center", zorder=5)

    # Legenda warna sesuai sketsa pengguna
    ax.plot([], [], color="#fca5a5", linewidth=2.5, label="Jalan Utama (Merah)")
    ax.plot([], [], color="#c084fc", linewidth=2.2, linestyle="--", label="Taman Kampus / Tanian (Ungu)")
    ax.plot([], [], color="#22c55e", linewidth=3.0, linestyle=":", label="Jalur Pintas Kecil/Jelek '>' (Hijau)")

    ax.set_title("Peta Graf Navigasi 25 Gedung UTM (Algoritma A*)", fontsize=12, fontweight="bold")
    ax.set_xlim(40, 830)
    ax.set_ylim(80, 930)
    ax.axis("off")
    ax.legend(loc="lower left", fontsize=8.5, frameon=True)
    return fig

# --- 4. ANTARMUKA PENGGUNA (STREAMLIT) ---
st.title("🗺️ UTM SmartNav: Pencarian Rute Terpendek Kampus")
st.caption("Implementasi Algoritma A* Search — 25 Gedung, Jalur Utama (Merah), Taman Kampus (Ungu), & Jalur Pintas (Hijau '>')")

node_options = {data["name"]: nid for nid, data in NODES.items()}

col1, col2, col3 = st.columns(3)
with col1:
    start_label = st.selectbox("📍 Lokasi Awal", list(node_options.keys()), index=0)
with col2:
    goal_label = st.selectbox("🏁 Lokasi Tujuan", list(node_options.keys()), index=4)
with col3:
    transport_mode = st.selectbox("🚶‍♂️ Moda Transportasi", list(SPEEDS.keys()))

allow_bad_shortcut = st.checkbox(
    "Izinkan melewati Jalur Pintas Kecil / Jalan Jelek (Garis Hijau '>') — lebih dekat namun kondisi jalan rusak/sempit",
    value=True,
)

if st.button("Cari Rute Terpendek", type="primary"):
    start_id = node_options[start_label]
    goal_id = node_options[goal_label]

    if start_id == goal_id:
        st.warning("Lokasi awal dan tujuan sama. Jarak tempuh: 0 meter.")
    else:
        mode_info = SPEEDS[transport_mode]
        graph = build_graph(mode_info, allow_bad_shortcut)
        path, total_distance, segments, trace_log = a_star_search(graph, start_id, goal_id)
        bfs_path, bfs_dist, bfs_expanded = bfs_comparison(graph, start_id, goal_id)

        if not path:
            st.error("Rute tidak ditemukan untuk kombinasi moda transportasi dan opsi jalur ini.")
        else:
            # Hitung total waktu dengan penalti kecepatan pada jalur hijau ('>')
            total_minutes = 0.0
            used_bad_shortcut = False
            for _, _, seg_dist, r_type in segments:
                if r_type == "shortcut_bad":
                    used_bad_shortcut = True
                    eff_speed = mode_info["speed"] * mode_info["bad_road_factor"]
                else:
                    eff_speed = mode_info["speed"]
                total_minutes += seg_dist / eff_speed

            minutes = int(total_minutes)
            seconds = int((total_minutes - minutes) * 60)

            route_names = [NODES[nid]["name"] for nid in path]
            st.success(f"**Rute Optimal A\*:** {' ➔ '.join(route_names)}")

            if used_bad_shortcut:
                st.warning("⚠️ Rute ini melewati **Jalur Pintas Kecil / Jalan Jelek (Garis Hijau `>`)**. Jarak lebih pendek, tetapi kecepatan melambat karena kondisi jalan kurang baik.")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Jarak (A*)", f"{total_distance:.0f} meter")
            m2.metric("Estimasi Waktu", f"{minutes} mnt {seconds} dtk")
            m3.metric("Simpul Dievaluasi A*", f"{len(trace_log)} titik")
            m4.metric("Simpul Dievaluasi BFS", f"{bfs_expanded} titik")

            st.pyplot(draw_campus_map(path))

            with st.expander("🔍 Lihat Proses Perhitungan Algoritma A* (g(n), h(n), f(n))"):
                st.write(
                    f"**Evaluasi Efisiensi:** **A\*** mengevaluasi **{len(trace_log)} simpul** (total jarak **{total_distance:.0f} m**), "
                    f"sedangkan **BFS** mengevaluasi **{bfs_expanded} simpul** (total jarak **{bfs_dist:.0f} m**)."
                )
                st.table(trace_log)