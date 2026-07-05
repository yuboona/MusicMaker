#!/usr/bin/env python3
"""
音域检测器 — 分析哼唱/人声的有效音域范围

功能：
1. 检测最低和最高有效音高
2. 识别最舒适音区（tessitura，剔除偶然的极限音）
3. 音域分类（男低/男中/男高/女低/女中/女高）
4. 钢琴键盘可视化
5. 支持多段录音对比

用法：
    python pitch_range.py <音频文件>
    python pitch_range.py <音频文件1> <音频文件2>   # 对比模式


────────── 技术分析思路 ──────────

┌─────────────────────────────────────────────────────────┐
│ 核心方法：piptrack + 分位数过滤                          │
│                                                         │
│ 1. piptrack 对每帧频谱做峰值检测，返回音高概率矩阵       │
│ 2. 用中位数 × 1.5 做阈值，筛掉气息和杂音                │
│ 3. 有效音高用 5%/95% 分位数定义"常规音域"，              │
│    用 1%/99% 分位数定义"极限音域"                        │
│                                                         │
│ 关键洞察：                                              │
│   - 5% 分位而不是 min → 排除偶然的低音（咳嗽/杂音）       │
│   - 95% 分位而不是 max → 排除偶然的尖峰（破音/哨音）      │
│   - 中段 50%（25%-75%）→ 最舒适音区（tessitura）          │
│                                                         │
│ tessitura（音区重心）比"能唱多高"更重要。                 │
│ 它是你在 10 分钟里能持续演唱的范围——                     │
│ 这才是编曲时真正有用的数字。                             │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ 音域分类 → MIDI 音高区间映射                             │
│                                                         │
│ 参考《新格罗夫音乐辞典》的标准音域分类：                  │
│   男低音 E2-E4   │  女低音 F3-F5                        │
│   男中音 A2-A4   │  女中音 A3-A5                        │
│   男高音 C3-C5   │  女高音 C4-C6                        │
│                                                         │
│ 哼唱通常比正式演唱窄一些，用模糊匹配而非硬边界。          │
└─────────────────────────────────────────────────────────┘
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 音域分类标准（MIDI 音符号码）
# 格式：(最低, 最高, 标签)
VOICE_CATEGORIES = [
    ((28, 52), "男低音 (Bass) — 深沉、有重量感"),
    ((33, 57), "男中音 (Baritone) — 温暖、有厚度"),
    ((36, 60), "男高音 (Tenor) — 明亮、有穿透力"),
    ((41, 65), "女低音 (Alto) — 醇厚、有磁性"),
    ((45, 69), "女中音 (Mezzo-soprano) — 柔美、有弹性"),
    ((48, 72), "女高音 (Soprano) — 清亮、有高度"),
]

# 钢琴键盘：白键的完整 octave 布局
# 每个八度: C D E F G A B
NOTE_ORDER = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_to_name(midi_num: int) -> str:
    """MIDI 编号 → 音名 + 八度（如 C4）"""
    octave = (midi_num // 12) - 1
    note_idx = midi_num % 12
    return f"{NOTE_ORDER[note_idx]}{octave}"


def detect_range(filepath: Path) -> dict:
    """
    分析单个音频的音域。
    返回：
        - conventional_range: 常规音域 (5%-95% 分位)
        - extreme_range: 极限音域 (1%-99% 分位)
        - tessitura: 最舒适音区 (25%-75% 分位)
        - voice_category: 音域分类
    """
    import numpy as np
    import librosa

    result = {
        "file": filepath.name,
        "duration": None,
        "conventional_range": None,
        "extreme_range": None,
        "tessitura": None,
        "voice_category": None,
    }

    y, sr = librosa.load(str(filepath), sr=None)
    result["duration"] = round(len(y) / sr, 1)

    # ---- 音高检测 ----
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

    # 过滤：只取能量高于中位数 × 1.5 的音高
    threshold = np.median(magnitudes) * 1.5
    valid_pitches = pitches[magnitudes > threshold]
    valid_pitches = valid_pitches[valid_pitches > 0]

    if len(valid_pitches) < 10:
        print(f"⚠️ 有效音高太少 ({len(valid_pitches)} 个)，可能录音质量不够")
        return result

    midi_pitches = librosa.hz_to_midi(valid_pitches)

    # 常规音域：5%-95% 分位（日常可用范围）
    p5 = int(np.percentile(midi_pitches, 5))
    p95 = int(np.percentile(midi_pitches, 95))
    result["conventional_range"] = (p5, p95)

    # 极限音域：1%-99% 分位（最宽范围，含极限音）
    p1 = int(np.percentile(midi_pitches, 1))
    p99 = int(np.percentile(midi_pitches, 99))
    result["extreme_range"] = (p1, p99)

    # 最舒适音区（tessitura）：25%-75% 分位
    p25 = int(np.percentile(midi_pitches, 25))
    p75 = int(np.percentile(midi_pitches, 75))
    result["tessitura"] = (p25, p75)

    # 音域分类：找最匹配的类别
    # 用常规音域的中点和跨度去匹配
    center = (p5 + p95) / 2
    span = p95 - p5
    best_score = -1

    for (cat_min, cat_max), label in VOICE_CATEGORIES:
        # 评分：中心接近 + 跨度接近
        cat_center = (cat_min + cat_max) / 2
        cat_span = cat_max - cat_min
        score = -abs(center - cat_center) * 0.6 - abs(span - cat_span) * 0.4
        if best_score is None or score > best_score:
            best_score = score
            result["voice_category"] = label

    return result


def piano_visualize(conventional: tuple, tessitura: tuple = None):
    """在 ASCII 钢琴键盘上标记音域"""
    # 键盘范围：C2 (36) 到 C6 (84)
    keyboard_start = 36
    keyboard_end = 84
    width = keyboard_end - keyboard_start + 1

    min_p, max_p = conventional

    # 构建键盘行
    white_keys = []
    black_keys = [" "] * (width * 2)  # 宽占位

    for midi in range(keyboard_start, keyboard_end + 1):
        note = NOTE_ORDER[midi % 12]
        pos = (midi - keyboard_start) * 2

        if "#" in note:  # 黑键
            marker = " "
            if min_p <= midi <= max_p:
                marker = "▪"
            black_keys[pos] = marker
        else:  # 白键
            marker = "│"
            if min_p <= midi <= max_p:
                marker = "█"
            # tessitura 用不同标记
            if tessitura and tessitura[0] <= midi <= tessitura[1]:
                marker = "▓"
            label = f"{note}{midi // 12 - 1}".ljust(3)
            white_keys.append(f"{marker}{label}")

    print()
    print("   " + "".join(white_keys))
    print()
    print("   █ = 常规音域 (5%-95%)   ▓ = 最舒适音区 (25%-75%)")
    print()


def compare_ranges(results: list[dict]):
    """对比多段录音的音域"""
    print()
    print("=" * 60)
    print("   📊 多段录音音域对比")
    print("=" * 60)
    print()
    print(f"   {'录音':<24} {'常规音域':<18} {'跨度':<8} {'最舒适区':<16}")
    print(f"   {'-'*24} {'-'*18} {'-'*8} {'-'*16}")

    for r in results:
        name = r["file"][:22]
        if r["conventional_range"]:
            lo, hi = r["conventional_range"]
            span = hi - lo
            cr = f"{midi_to_name(lo)} → {midi_to_name(hi)}"
            cr_str = f"{cr:<18} {span} 半音"
        else:
            cr_str = "检测失败"

        if r["tessitura"]:
            lo, hi = r["tessitura"]
            te = f"{midi_to_name(lo)} → {midi_to_name(hi)}"
        else:
            te = "-"

        print(f"   {name:<24} {cr_str:<22} {te:<16}")

    # 合并音域：所有录音的最大范围
    all_lows = []
    all_highs = []
    for r in results:
        if r["conventional_range"]:
            all_lows.append(r["conventional_range"][0])
            all_highs.append(r["conventional_range"][1])

    if all_lows and all_highs:
        combined = (min(all_lows), max(all_highs))
        print()
        print(f"   📐 合并音域: {midi_to_name(combined[0])} → {midi_to_name(combined[1])}")
        print(f"      (跨度 {combined[1] - combined[0]} 半音 = {combined[1] - combined[0]:.1f} 个八度)")
        print(f"      → 编曲时，主旋律不要超出这个范围")

    print()


def print_report(result: dict):
    """打印单个录音的音域报告"""
    print()
    print("=" * 60)
    print(f"   🎤 音域分析: {result['file']}")
    print("=" * 60)
    print(f"   时长: {result['duration']} 秒")

    if not result["conventional_range"]:
        print("   ❌ 未能检测到有效音高")
        return

    lo, hi = result["conventional_range"]
    span = hi - lo

    print()
    print("📏 常规音域（日常可用）")
    print(f"   最低: {midi_to_name(lo)}  ({lo} MIDI)")
    print(f"   最高: {midi_to_name(hi)} ({hi} MIDI)")
    print(f"   跨度: {span} 半音 ({span / 12:.1f} 个八度)")

    if result["tessitura"]:
        tlo, thi = result["tessitura"]
        print()
        print("🎯 最舒适音区（Tessitura）")
        print(f"   {midi_to_name(tlo)} → {midi_to_name(thi)}")
        print(f"   ↑ 编曲时旋律主要放在这个区间，最自然、最不费力")

    if result["extreme_range"] and result["extreme_range"] != result["conventional_range"]:
        elo, ehi = result["extreme_range"]
        if elo != lo or ehi != hi:
            print()
            print("🔺 极限音域（含偶然音，偶尔可达）")
            print(f"   {midi_to_name(elo)} → {midi_to_name(ehi)}")

    if result["voice_category"]:
        print()
        print("🗂️ 音域分类")
        print(f"   {result['voice_category']}")

    # 钢琴键盘可视化
    tess = result.get("tessitura")
    piano_visualize(result["conventional_range"], tess)

    # 给编曲的建议
    print("💡 编曲建议")
    if span < 12:
        print("   音域较窄（<1 个八度），建议用简单的重复动机而非大跳跃旋律")
    elif span < 24:
        print(f"   音域适中（{span} 半音），可以设计有起伏的旋律线")
    else:
        print(f"   音域宽（{span} 半音），可以有大跨度的情绪对比")

    if result["tessitura"]:
        tlo, thi = result["tessitura"]
        print(f"   主旋律集中在 {midi_to_name(tlo)}-{midi_to_name(thi)} 写，偶尔跳出去制造高点")
    print()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    filepaths = []
    for arg in sys.argv[1:]:
        fp = Path(arg)
        if not fp.is_absolute():
            fp = PROJECT_ROOT / fp
        if not fp.exists():
            print(f"❌ 文件不存在: {fp}")
            sys.exit(1)
        filepaths.append(fp)

    results = []
    for fp in filepaths:
        result = detect_range(fp)
        results.append(result)
        print_report(result)

    # 多段录音时打印对比
    if len(results) >= 2:
        compare_ranges(results)


if __name__ == "__main__":
    main()
