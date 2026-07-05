#!/usr/bin/env python3
"""
灵感库管理器 — 扫描和管理你的哼唱录音库

功能：
1. 扫描 01_Humming_Ideas/ 中的新录音
2. 自动提取文件名中的日期和情绪标签
3. 用 librosa 分析：时长、BPM、调性估计、音域
4. 生成 Markdown 格式的灵感库索引
5. 对比两段录音的调性和节奏兼容性

用法：
    python manage_library.py scan          # 扫描灵感库，生成索引
    python manage_library.py compare A B   # 对比两段录音的兼容性
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# ============================================================
# 配置
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
HUMMING_DIR = PROJECT_ROOT / "01_Humming_Ideas"
OUTPUT_INDEX = PROJECT_ROOT / "01_Humming_Ideas" / "INDEX.md"


def scan_library() -> list[dict]:
    """扫描灵感库目录，返回所有录音文件的信息"""
    recordings = []

    if not HUMMING_DIR.exists():
        print(f"❌ 灵感库目录不存在: {HUMMING_DIR}")
        return recordings

    for audio_file in sorted(HUMMING_DIR.rglob("*.mp3")):
        # 解析文件名: 日期_情绪_秒数.mp3 或 MMDD_描述_XXs.mp3
        filename = audio_file.stem
        parts = filename.split("_")

        info = {
            "path": str(audio_file.relative_to(PROJECT_ROOT)),
            "filename": audio_file.name,
            "date": None,
            "mood": "未标注",
            "duration_label": "未知",
            "duration_sec": None,
            "bpm": None,
            "key": None,
        }

        # 尝试解析日期
        if len(parts) >= 1 and len(parts[0]) == 4 and parts[0].isdigit():
            try:
                month, day = parts[0][:2], parts[0][2:]
                year = datetime.now().year  # 默认当年
                info["date"] = f"{year}-{month}-{day}"
            except ValueError:
                info["date"] = parts[0]

        # 尝试解析情绪
        if len(parts) >= 2:
            info["mood"] = parts[1]

        # 尝试解析时长
        if len(parts) >= 3:
            dur_str = parts[2].replace("s", "").replace("S", "")
            if dur_str.isdigit():
                info["duration_label"] = f"{dur_str}秒"

        recordings.append(info)

    return recordings


def analyze_audio(filepath: Path) -> dict:
    """分析单个音频文件，提取 BPM 和调性信息"""
    result = {"bpm": None, "key": None, "duration_sec": None}

    try:
        import librosa
        import numpy as np

        y, sr = librosa.load(str(filepath), sr=None)
        result["duration_sec"] = round(len(y) / sr, 1)

        # BPM 估计
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        if hasattr(tempo, 'item'):
            result["bpm"] = round(float(tempo.item()))
        else:
            result["bpm"] = round(float(tempo))

        # 调性估计 (Krumhansl-Schmuckler 算法)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)

        # 大调和小调的 Krumhansl-Kessler 模板
        major_profile = np.array(
            [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
        )
        minor_profile = np.array(
            [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
        )

        note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

        # 计算与每个调的相关性
        best_corr = -999
        best_key = "unknown"

        for i in range(12):
            rotated = np.roll(chroma_mean, i)
            corr_major = np.corrcoef(rotated, major_profile)[0, 1]
            corr_minor = np.corrcoef(rotated, minor_profile)[0, 1]

            if corr_major > best_corr:
                best_corr = corr_major
                # 调 = 移位的音名，大调
                tonic_idx = (-i) % 12
                best_key = f"{note_names[tonic_idx]} 大调"

            if corr_minor > best_corr:
                best_corr = corr_minor
                tonic_idx = (-i) % 12
                best_key = f"{note_names[tonic_idx]} 小调"

        result["key"] = best_key if best_corr > 0.5 else "不确定"

    except ImportError:
        print("⚠️  librosa 未安装，跳过音频分析。安装: pip install librosa")
    except Exception as e:
        print(f"⚠️  分析 {filepath.name} 时出错: {e}")

    return result


def generate_index(recordings: list[dict]):
    """生成 Markdown 格式的灵感库索引"""
    lines = [
        "# 🎤 灵感库索引",
        "",
        f"> 自动生成于 {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"> 共 {len(recordings)} 段录音",
        "",
        "## 录音列表",
        "",
        "| # | 日期 | 情绪标签 | 时长 | BPM | 调性 | 文件名 |",
        "|---|------|---------|------|-----|------|--------|",
    ]

    for i, rec in enumerate(recordings, 1):
        date = rec.get("date", "?")
        mood = rec.get("mood", "未标注")
        dur = rec.get("duration_label", "?")
        bpm = str(rec.get("bpm", "?")) if rec.get("bpm") else "?"
        key = rec.get("key", "?") or "?"
        filename = rec.get("filename", "?")

        lines.append(f"| {i} | {date} | {mood} | {dur} | {bpm} | {key} | {filename} |")

    lines.extend([
        "",
        "## 情绪分布",
        "",
    ])

    # 统计情绪标签
    mood_counts = {}
    for rec in recordings:
        mood = rec.get("mood", "未标注")
        mood_counts[mood] = mood_counts.get(mood, 0) + 1

    for mood, count in sorted(mood_counts.items(), key=lambda x: -x[1]):
        bar = "█" * count
        lines.append(f"- **{mood}**: {bar} ({count})")

    lines.extend([
        "",
        "---",
        "",
        "💡 **提示**: 想要分析某段录音的详细信息，告诉 Claude Code：",
        '`"分析 01_Humming_Ideas/2026-07/0705_夜晚漂浮_23s.mp3"`',
        "",
    ])

    OUTPUT_INDEX.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_INDEX.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ 灵感库索引已生成: {OUTPUT_INDEX}")


def compare_recordings(path_a: str, path_b: str):
    """对比两段录音的兼容性"""
    file_a = PROJECT_ROOT / path_a
    file_b = PROJECT_ROOT / path_b

    if not file_a.exists():
        print(f"❌ 文件不存在: {file_a}")
        return
    if not file_b.exists():
        print(f"❌ 文件不存在: {file_b}")
        return

    print(f"📊 对比分析:")
    print(f"  A: {file_a.name}")
    print(f"  B: {file_b.name}")
    print()

    info_a = analyze_audio(file_a)
    info_b = analyze_audio(file_b)

    print(f"  A: BPM={info_a['bpm']}, 调性={info_a['key']}, 时长={info_a['duration_sec']}s")
    print(f"  B: BPM={info_b['bpm']}, 调性={info_b['key']}, 时长={info_b['duration_sec']}s")
    print()

    # 兼容性判断
    issues = []
    if info_a["bpm"] and info_b["bpm"]:
        bpm_diff = abs(info_a["bpm"] - info_b["bpm"])
        if bpm_diff <= 5:
            print(f"✅ BPM 接近（差 {bpm_diff}），节奏兼容")
        elif bpm_diff <= 15:
            print(f"⚠️  BPM 有差距（差 {bpm_diff}），可以调整但需要刻意处理")
            issues.append("节奏差异")
        else:
            print(f"❌ BPM 差距大（差 {bpm_diff}），可能不适合作为同一首歌")
            issues.append("节奏差异大")

    if info_a["key"] and info_b["key"] and "不确定" not in [info_a["key"], info_b["key"]]:
        key_a_root = info_a["key"].split()[0]
        key_b_root = info_b["key"].split()[0]
        if key_a_root == key_b_root:
            print(f"✅ 调性相同（{info_a['key']}），直接兼容")
        else:
            # 检查是否为关系大小调
            major_minor_pairs = {
                "C": "A", "G": "E", "D": "B", "A": "F#", "E": "C#",
                "B": "G#", "F#": "D#", "F": "D", "Bb": "G", "Eb": "C", "Ab": "F",
            }
            is_relative = False
            for maj, rel_min in major_minor_pairs.items():
                if (key_a_root == maj and key_b_root == rel_min) or \
                   (key_b_root == maj and key_a_root == rel_min):
                    is_relative = True
                    break

            if is_relative:
                print(f"🟡 调性为关系大小调（{info_a['key']} ↔ {info_b['key']}），可以共存但有情绪差异")
            else:
                print(f"⚠️  调性不同（{info_a['key']} vs {info_b['key']}），可以通过转调统一")
                issues.append("调性差异")

    if not issues:
        print()
        print("🎉 结论：这两段灵感兼容性良好，可以考虑作为同一首歌的不同段落！")
    else:
        print()
        print(f"📋 需要处理的问题: {', '.join(issues)}")
        print("💡 建议：告诉 Claude Code 你的偏好，它可以帮你写转调脚本或调整 BPM。")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    command = sys.argv[1]

    if command == "scan":
        recordings = scan_library()
        if not recordings:
            print("📭 灵感库为空。请把哼唱录音放入 01_Humming_Ideas/ 目录。")
            print("   命名格式: 日期_情绪_秒数.mp3（如 0705_夜晚漂浮_23s.mp3）")
            return

        # 分析每个录音
        print(f"🔍 正在分析 {len(recordings)} 段录音...")
        for rec in recordings:
            filepath = PROJECT_ROOT / rec["path"]
            result = analyze_audio(filepath)
            rec.update(result)
            status = f"BPM={rec['bpm']}, Key={rec['key']}" if rec["bpm"] else "分析失败"
            print(f"  ✓ {rec['filename']} → {status}")

        generate_index(recordings)

    elif command == "compare":
        if len(sys.argv) < 4:
            print("用法: python manage_library.py compare <文件A路径> <文件B路径>")
            print("例如: python manage_library.py compare 01_Humming_Ideas/2026-07/0705_a_23s.mp3 01_Humming_Ideas/2026-07/0705_b_15s.mp3")
            return
        compare_recordings(sys.argv[2], sys.argv[3])

    else:
        print(f"未知命令: {command}")
        print(__doc__)


if __name__ == "__main__":
    main()
