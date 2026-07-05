#!/usr/bin/env python3
"""
音频分析器 — 深入分析哼唱录音的音乐特征

功能：
1. 调性检测（Krumhansl-Schmuckler 算法）
2. BPM/速度检测
3. 音域范围分析
4. 频谱特征分析
5. 风格方向推荐

用法：
    python analyze_audio.py <音频文件路径>
    python analyze_audio.py 01_Humming_Ideas/2026-07/0705_夜晚漂浮_23s.mp3


────────── 技术分析思路 ──────────

整个分析管线分为 5 步，每步选用的方法和理由如下：

┌─────────────────────────────────────────────────────────┐
│ 1. BPM 检测 → onset_strength + beat_track              │
│                                                         │
│ 先用 onset_strength 算出每帧的"能量爆发点"（onset），     │
│ 再通过动态规划找到最可能的拍点序列。                        │
│ 人对速度的直觉来自重音间隔，onset 就是这个间隔的数学表达。  │
│                                                         │
│ 选型理由：动态规划比简单自相关更稳定，能处理哼唱中          │
│ 节奏不规整的情况。                                        │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ 2. 音高检测 → piptrack (pitch saliency)                │
│                                                         │
│ 对每帧频谱做峰值检测，取能量最强的几个频率分量，            │
│ 通过 saliency 函数将频率映射为音高概率。                    │
│ 然后用 5% 和 95% 分位数剔除哼唱中偶然的杂音和气息。       │
│                                                         │
│ 选型理由：piptrack 不需要训练数据，直接对频谱做              │
│ 抛物线插值，适合单旋律（哼唱）场景。                       │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ 3. 调性检测 → chroma_cqt + Krumhansl-Schmuckler 模板    │
│                                                         │
│ 用 CQT（常数 Q 变换）提取 12 维 chroma 向量，每维对应      │
│ 一个音高类别（C/C#/D/...），取时间均值得到整曲的           │
│ 音高分布。然后将这个分布与已知的大调/小调模板               │
│（K-K profiles）做皮尔逊相关，取最高分。                    │
│                                                         │
│ 选型理由：CQT 在低频分辨率优于 STFT，更适合哼唱这种          │
│ 基频较低的场景。K-K 模板是音乐心理学实验得出的，             │
│ 不是纯数学产物——它反映的是人脑对调性中心的实际感知。        │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ 4. 频谱质心 → spectral_centroid                         │
│                                                         │
│ 频谱质心 = 各频率分量按能量加权的平均频率。                │
│ 物理含义：声音的重心偏向哪里。                             │
│ >2000Hz → 明亮的咝音成分多（气息、齿音）                   │
│ <1000Hz → 温暖、暗沉（胸腔共鸣为主）                       │
│                                                         │
│ 用途：辅助判断哼唱者的音色特征，用于风格匹配。              │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ 5. 风格推荐 → 规则引擎（BPM 区间 + 调性模式）             │
│                                                         │
│ 不是 ML，是显式规则。每种风格预定义了 BPM 范围和调性偏好。  │
│ 好处：透明、可解释、用户可直接修改规则（修改 STYLE_        │
│ INSTRUMENTS 字典即可）。                                  │
│                                                         │
│ 缺陷：只能区分大类，无法做精细风格判断。如果需要更准         │
│ 确的风格分类，可以后续接入预训练模型（如 genre classifier）。│
└─────────────────────────────────────────────────────────┘
"""

import sys
from pathlib import Path

# ============================================================
# 配置
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 和弦进行推荐库：情绪 → [(和弦进行, 参考歌曲, 适合风格)]
CHORD_RECOMMENDATIONS = {
    "温暖": [
        ("I - V - vi - IV (C - G - Am - F)", "Let It Be, Don't Stop Believin'", "流行/民谣"),
        ("I - IV - V - I (C - F - G - C)", "Twist and Shout", "摇滚/流行"),
    ],
    "忧伤": [
        ("vi - IV - I - V (Am - F - C - G)", "Someone Like You", "流行/抒情"),
        ("i - VI - III - VII (Am - F - C - G 小调版)", "Hello", "抒情/力量"),
    ],
    "梦幻": [
        ("I - iii - vi - IV (C - Em - Am - F)", "Space Oddity", "梦幻流行/太空"),
        ("IV - I - V - vi (F - C - G - Am)", "Clocks", "独立/氛围"),
    ],
    "力量/希望": [
        ("I - V - vi - IV (C - G - Am - F)", "Let It Be", "流行/摇滚"),
        ("vi - V - IV - V (Am - G - F - G)", "Eye of the Tiger", "摇滚/史诗"),
    ],
    "复古/怀旧": [
        ("I - vi - ii - V (C - Am - Dm - G)", "Fly Me to the Moon", "爵士/复古流行"),
        ("I - IV - I - V (C - F - C - G)", "Can't Help Falling in Love", "抒情/怀旧"),
    ],
    "内省/沉思": [
        ("i - bVI - bIII - bVII (Am - F - C - G)", "Mad World", "氛围/另类"),
        ("i - iv - bVI - V (Am - Dm - F - E)", "House of the Rising Sun", "民谣/蓝调"),
    ],
}

# 风格配置：风格 → 推荐的乐器组合
STYLE_INSTRUMENTS = {
    "流行": {
        "地基": "电贝斯 + 底鼓（简洁有力）",
        "骨架": "钢琴（中频为主）",
        "血肉": "明亮合成器 Lead 或钢琴高音区",
        "灵魂": "弦乐 Pad + 镲片（副歌进入）",
        "bpm_range": (80, 130),
        "参考工具": "BandLab 自带钢琴 + 鼓组 Loop 即可覆盖",
    },
    "民谣": {
        "地基": "贝斯（轻声）+ 轻拍鼓或沙锤",
        "骨架": "原声吉他（指弹或扫弦）",
        "血肉": "口琴/小提琴/钢琴高音",
        "灵魂": "弦乐轻铺底 + 自然环境音效",
        "bpm_range": (70, 100),
        "参考工具": "Ample Guitar M Lite + Spitfire LABS Strings",
    },
    "电子/氛围": {
        "地基": "合成器贝斯 + 电子鼓（包含 808）",
        "骨架": "合成器 Pad（持续和弦）",
        "血肉": "合成器 Lead/Arp（琶音）",
        "灵魂": "音效 + 大混响 + 滤波器扫频",
        "bpm_range": (90, 140),
        "参考工具": "Vital 合成器 + BandLab 电子 Loop",
    },
    "R&B/灵魂": {
        "地基": "电贝斯（注重律动）+ 碎拍/摇摆鼓",
        "骨架": "电钢琴/Rhodes（温暖中频）",
        "血肉": "人声和声（或 MIDI 替代）+ 键盘 solo",
        "灵魂": "Warm Pad + 镲片（碎拍感）",
        "bpm_range": (65, 95),
        "参考工具": "Spitfire LABS Soft Piano + BandLab 鼓组（选碎拍）",
    },
    "Indie/独立": {
        "地基": "贝斯 + Lo-fi 鼓/轻力度鼓",
        "骨架": "吉他或钢琴（可以故意不完美）",
        "血肉": "特殊音色（钟琴、音乐盒、老式合成器）",
        "灵魂": "低保真效果 + 磁带噪音 + 自然混响",
        "bpm_range": (75, 120),
        "参考工具": "Spitfire LABS（多样音色库）+ Ample Guitar M Lite",
    },
}


def analyze_audio(filepath: Path) -> dict:
    """全面的音频分析"""
    result = {
        "duration_sec": None,
        "bpm": None,
        "key": None,
        "key_confidence": None,
        "pitch_range": None,
        "spectral_centroid": None,
        "rms_energy": None,
    }

    try:
        import librosa
        import numpy as np

        print("🔄 加载音频文件...")
        y, sr = librosa.load(str(filepath), sr=None)
        duration = len(y) / sr
        result["duration_sec"] = round(duration, 1)
        print(f"   ✓ 时长: {result['duration_sec']} 秒, 采样率: {sr}Hz")

        # ---- BPM ----
        print("🔄 检测速度...")
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        # librosa 返回的是 numpy 数组，取第一个元素
        if hasattr(tempo, 'item'):
            result["bpm"] = round(float(tempo.item()))
        else:
            result["bpm"] = round(float(tempo))
        print(f"   ✓ BPM: {result['bpm']}")

        # ---- 音高检测 ----
        print("🔄 分析音高...")
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

        # 提取有效音高（magnitude > threshold）
        threshold = np.median(magnitudes) * 1.5
        valid_pitches = pitches[magnitudes > threshold]
        valid_pitches = valid_pitches[valid_pitches > 0]

        if len(valid_pitches) > 0:
            midi_pitches = librosa.hz_to_midi(valid_pitches)
            min_midi = int(np.percentile(midi_pitches, 5))
            max_midi = int(np.percentile(midi_pitches, 95))
            result["pitch_range"] = {
                "min_hz": round(float(librosa.midi_to_hz(min_midi)), 1),
                "max_hz": round(float(librosa.midi_to_hz(max_midi)), 1),
                "min_note": librosa.midi_to_note(min_midi),
                "max_note": librosa.midi_to_note(max_midi),
            }
            print(f"   ✓ 音域: {result['pitch_range']['min_note']} → {result['pitch_range']['max_note']}")
        else:
            print("   ⚠️  未检测到有效音高")

        # ---- 调性检测 ----
        # 原理：提取 12 维 chroma（音高类别分布），与 Krumhansl-Kessler 模板做相关
        #
        # K-K 模板的来源（1986，认知心理学实验）：
        #   1. 让被试听一个调性上下文（如 C 大调音阶）
        #   2. 随机播 12 个音之一，被试打分"这个音在当前调性里有多合适？"(1-7)
        #   3. 收集大量评分 → 得到每个音在该调性里的"归属感"权重
        #
        #   C 大调: C=6.35(主音最稳), G=5.19(五级第二稳), E=4.38(三级定义大小调)
        #           这些数字是人的心理感知，不是数学推导——所以调性检测本质上是
        #           "和几千人的平均听觉做对比"
        #
        # 为什么用皮尔逊相关而不是简单的模板匹配？
        #   相关衡量的是分布形状的相似度，能容忍整体音量的差异。
        #   你哼得大声小声，chroma 的绝对大小会变，但 12 维的相对形状不变。
        print("🔄 检测调性...")
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)

        major_profile = np.array(
            [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
        )
        minor_profile = np.array(
            [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
        )
        note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

        best_corr = -999
        best_key = "unknown"
        best_mode = ""

        for i in range(12):
            rotated = np.roll(chroma_mean, i)
            corr_major = np.corrcoef(rotated, major_profile)[0, 1]
            corr_minor = np.corrcoef(rotated, minor_profile)[0, 1]

            if corr_major > best_corr:
                best_corr = corr_major
                tonic_idx = (-i) % 12
                best_key = f"{note_names[tonic_idx]} 大调"
                best_mode = "major"

            if corr_minor > best_corr:
                best_corr = corr_minor
                tonic_idx = (-i) % 12
                best_key = f"{note_names[tonic_idx]} 小调"
                best_mode = "minor"

        result["key"] = best_key if best_corr > 0.5 else "不确定"
        result["key_confidence"] = round(float(best_corr), 3)
        conf_label = "高" if best_corr > 0.7 else ("中" if best_corr > 0.5 else "低")
        print(f"   ✓ 调性: {result['key']}（置信度: {conf_label}, {best_corr:.3f}）")

        # ---- 频谱特征 ----
        print("🔄 分析频谱特征...")
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        result["spectral_centroid"] = round(float(np.mean(spectral_centroid)), 0)
        brightness = "明亮" if result["spectral_centroid"] > 2000 else ("温暖" if result["spectral_centroid"] > 1000 else "暗沉")
        print(f"   ✓ 频谱质心: {result['spectral_centroid']}Hz → 听感: {brightness}")

        # ---- 能量 ----
        rms = librosa.feature.rms(y=y)[0]
        result["rms_energy"] = round(float(np.mean(rms)), 4)
        energy = "高" if result["rms_energy"] > 0.1 else ("中" if result["rms_energy"] > 0.03 else "低")
        print(f"   ✓ 能量水平: {energy} ({result['rms_energy']:.4f})")

    except ImportError:
        print("❌ 需要安装 librosa: pip install librosa")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 分析出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    return result


def recommend_style(result: dict) -> list[dict]:
    """根据分析结果推荐风格方向"""
    recommendations = []
    bpm = result.get("bpm")
    key = result.get("key", "")
    is_minor = "小调" in key if key else False

    for style, config in STYLE_INSTRUMENTS.items():
        score = 0
        reasons = []

        # BPM 匹配
        bpm_min, bpm_max = config["bpm_range"]
        if bpm and bpm_min <= bpm <= bpm_max:
            score += 2
            reasons.append(f"BPM {bpm} 在 {style} 的典型范围内")
        elif bpm:
            if bpm < bpm_min:
                reasons.append(f"BPM 偏低（{bpm}），可能需要加快速度")
            else:
                reasons.append(f"BPM 偏高（{bpm}），可能需要放慢速度")

        # 调性匹配
        if is_minor and style in ["忧伤", "内省/沉思", "R&B/灵魂"]:
            score += 1
            reasons.append("小调天然适合有深度/忧伤的表达")
        elif not is_minor and style in ["温暖", "力量/希望", "流行"]:
            score += 1
            reasons.append("大调适合明亮/积极的表达")

        if score > 0:
            recommendations.append({
                "style": style,
                "score": score,
                "reasons": reasons,
                "instruments": config,
            })

    recommendations.sort(key=lambda x: -x["score"])
    return recommendations


def print_report(result: dict, filepath: Path):
    """打印格式化的分析报告"""
    print()
    print("=" * 60)
    print(f"   🎵 音频分析报告: {filepath.name}")
    print("=" * 60)
    print()
    print("📊 基础数据")
    print(f"   时长:    {result.get('duration_sec', '?')} 秒")
    print(f"   速度:    {result.get('bpm', '?')} BPM")
    print(f"   调性:    {result.get('key', '?')}")
    if result.get("key_confidence"):
        conf = result["key_confidence"]
        bar = "█" * int(conf * 10) + "░" * (10 - int(conf * 10))
        print(f"   置信度:  [{bar}] {conf:.3f}")

    if result.get("pitch_range"):
        pr = result["pitch_range"]
        print()
        print("🎤 音域")
        print(f"   范围:    {pr['min_note']} ({pr['min_hz']}Hz) → {pr['max_note']} ({pr['max_hz']}Hz)")

    if result.get("spectral_centroid"):
        print()
        print("🎨 音色特征")
        brightness = "明亮" if result["spectral_centroid"] > 2000 else ("温暖" if result["spectral_centroid"] > 1000 else "暗沉")
        print(f"   频谱质心: {result['spectral_centroid']}Hz")
        print(f"   听感:     {brightness}")

    print()
    print("🎯 风格推荐")
    print("-" * 40)
    recommendations = recommend_style(result)
    if recommendations:
        for i, rec in enumerate(recommendations[:3], 1):
            star = "⭐" * rec["score"]
            print(f"\n  选项 {i}: {rec['style']} {star}")
            for reason in rec["reasons"]:
                print(f"    → {reason}")
            inst = rec["instruments"]
            print(f"    🥁 地基: {inst['地基']}")
            print(f"    🎹 骨架: {inst['骨架']}")
            print(f"    🎻 血肉: {inst['血肉']}")
            print(f"    ✨ 灵魂: {inst['灵魂']}")
    else:
        print("   （风格推荐需要更多数据，请尝试用更清晰的哼唱录音）")

    print()
    print("🎹 和弦进行推荐")
    print("-" * 40)

    # 根据调性模式推荐
    is_minor = "小调" in result.get("key", "")
    if is_minor:
        mood_keys = ["忧伤", "内省/沉思", "梦幻"]
    else:
        mood_keys = ["温暖", "力量/希望", "复古/怀旧"]

    shown = set()
    for mood_key in mood_keys:
        if mood_key in CHORD_RECOMMENDATIONS:
            for chord, song, style in CHORD_RECOMMENDATIONS[mood_key]:
                if chord not in shown:
                    shown.add(chord)
                    print(f"   {chord}")
                    print(f"   → 参考: {song} | 适合: {style}")
                    print()

    print("=" * 60)
    print()
    print("💡 下一步:")
    print("   1. 告诉 Claude Code 你喜欢的风格方向（选项 1/2/3）")
    print("   2. Claude Code 会帮你选择合适的和弦进行和乐器配置")
    print("   3. 在 BandLab 中开始搭建第一版 Demo")
    print()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("用法: python analyze_audio.py <音频文件路径>")
        sys.exit(1)

    filepath = Path(sys.argv[1])
    if not filepath.is_absolute():
        filepath = PROJECT_ROOT / filepath

    if not filepath.exists():
        print(f"❌ 文件不存在: {filepath}")
        sys.exit(1)

    result = analyze_audio(filepath)
    print_report(result, filepath)


if __name__ == "__main__":
    main()
