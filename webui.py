import os
import random
import shutil
import sys
import traceback
import gc
import html
import base64
from datetime import datetime
from pathlib import Path
from typing import Literal, Tuple

import numpy as np
import torch
import librosa
import soundfile as sf
import gradio as gr

from preprocess.pipeline import PreprocessPipeline
from soulxsinger.utils.file_utils import load_config
from cli.inference import build_model as build_svs_model, process as svs_process


ROOT = Path(__file__).parent
GSAP_SOURCE = ""  # Optional animation vendor asset is excluded from this archive.
SAMPLE_RATE = 44100
PROMPT_MAX_SEC_DEFAULT = 30
TARGET_MAX_SEC_DEFAULT = 60
PROMPT_MAX_MERGE_DURATION_MS = 30000
TARGET_MAX_MERGE_DURATION_MS = 60000

ENGLISH_EXAMPLE_PROMPT_AUDIO = "example/audio/en_prompt.mp3"
ENGLISH_EXAMPLE_PROMPT_META = "example/audio/en_prompt.json"
ENGLISH_EXAMPLE_TARGET_AUDIO = "example/audio/en_target.mp3"
ENGLISH_EXAMPLE_TARGET_META = "example/audio/en_target.json"

MANDARIN_EXAMPLE_PROMPT_AUDIO = "example/audio/zh_prompt.mp3"
MANDARIN_EXAMPLE_PROMPT_META = "example/audio/zh_prompt.json"
MANDARIN_EXAMPLE_TARGET_AUDIO = "example/audio/zh_target.mp3"
MANDARIN_EXAMPLE_TARGET_META = "example/audio/zh_target.json"

CANTONESE_EXAMPLE_PROMPT_AUDIO = "example/audio/yue_prompt.mp3"
CANTONESE_EXAMPLE_PROMPT_META = "example/audio/yue_prompt.json"
CANTONESE_EXAMPLE_TARGET_AUDIO = "example/audio/yue_target.mp3"
CANTONESE_EXAMPLE_TARGET_META = "example/audio/yue_target.json"

MUSIC_EXAMPLE_TARGET_AUDIO = "example/audio/music.mp3"
MUSIC_EXAMPLE_TARGET_META = "example/audio/music.json"

# Use absolute paths so examples can hydrate the internal metadata state.
EXAMPLES_LIST = [
    [
        str(ROOT / MANDARIN_EXAMPLE_PROMPT_AUDIO),
        str(ROOT / MANDARIN_EXAMPLE_TARGET_AUDIO),
        str(ROOT / MANDARIN_EXAMPLE_PROMPT_META),
        str(ROOT / MANDARIN_EXAMPLE_TARGET_META),
        "Mandarin",
        "Mandarin",
        "melody-controlled",
        False,
        True,
        True,
        0,
    ],
    [
        str(ROOT / MANDARIN_EXAMPLE_PROMPT_AUDIO),
        str(ROOT / CANTONESE_EXAMPLE_TARGET_AUDIO),
        str(ROOT / MANDARIN_EXAMPLE_PROMPT_META),
        str(ROOT / CANTONESE_EXAMPLE_TARGET_META),
        "Mandarin",
        "Cantonese",
        "melody-controlled",
        False,
        True,
        True,
        0,
    ],
    [
        str(ROOT / MANDARIN_EXAMPLE_PROMPT_AUDIO),
        str(ROOT / ENGLISH_EXAMPLE_TARGET_AUDIO),
        str(ROOT / MANDARIN_EXAMPLE_PROMPT_META),
        str(ROOT / ENGLISH_EXAMPLE_TARGET_META),
        "Mandarin",
        "English",
        "melody-controlled",
        False,
        True,
        True,
        0,
    ],
    [
        str(ROOT / MANDARIN_EXAMPLE_PROMPT_AUDIO),
        str(ROOT / MUSIC_EXAMPLE_TARGET_AUDIO),
        str(ROOT / MANDARIN_EXAMPLE_PROMPT_META),
        str(ROOT / MUSIC_EXAMPLE_TARGET_META),
        "Mandarin",
        "Mandarin",
        "score-controlled",
        False,
        True,
        True,
        0,
    ],
]


# i18n
_I18N_KEY2LANG = dict(
    display_lang_label=dict(en="Display Language", zh="显示语言"),
    service_ready=dict(en="Model ready", zh="模型已就绪"),
    sources_title=dict(en="Choose the voices", zh="选择声音素材"),
    sources_desc=dict(en="A short reference sets the singer; the target supplies melody and lyrics.", zh="参考音频决定音色，目标音频提供旋律与歌词。"),
    prompt_title=dict(en="Reference voice", zh="参考音色"),
    prompt_desc=dict(en=f"Use a clear vocal, up to {PROMPT_MAX_SEC_DEFAULT} seconds.", zh=f"建议使用清晰干净的人声，最长 {PROMPT_MAX_SEC_DEFAULT} 秒。"),
    target_title=dict(en="Target song", zh="目标歌曲"),
    target_desc=dict(en=f"Provides melody and lyrics, up to {TARGET_MAX_SEC_DEFAULT} seconds.", zh=f"提供旋律和歌词，最长 {TARGET_MAX_SEC_DEFAULT} 秒。"),
    settings_title=dict(en="Generation settings", zh="生成设置"),
    result_title=dict(en="Generated performance", zh="生成结果"),
    seed_label=dict(en="Seed", zh="种子"),
    prompt_audio_label=dict(en="Reference audio", zh="参考音频"),
    target_audio_label=dict(en="Target audio", zh="目标音频"),
    synthesis_btn_label=dict(en="Generate singing voice", zh="开始生成"),
    retry_label=dict(en="Process again", zh="重新处理"),
    generated_audio_label=dict(en="Result audio", zh="结果音频"),
    prompt_lyric_lang_label=dict(en="Reference language", zh="参考歌词语种"),
    target_lyric_lang_label=dict(en="Target language", zh="目标歌词语种"),
    lyric_lang_mandarin=dict(en="Mandarin", zh="普通话"),
    lyric_lang_cantonese=dict(en="Cantonese", zh="粤语"),
    lyric_lang_english=dict(en="English", zh="英语"),
    warn_missing_synthesis=dict(
        en="Wait until both audio inputs have finished processing.",
        zh="请等待两段音频处理完成后再生成。",
    ),
    prompt_vocal_sep_label=dict(en="Separate backing track", zh="分离伴奏"),
    target_vocal_sep_label=dict(en="Separate backing track", zh="分离伴奏"),
    option_yes=dict(en="yes", zh="是"),
    option_no=dict(en="no", zh="否"),
    auto_shift_label=dict(en="Auto pitch shift", zh="自动变调"),
    pitch_shift_label=dict(en="Pitch shift (semitones)", zh="指定变调（半音）"),
    control_type_label=dict(en="Control type", zh="控制类型"),
    control_melody=dict(en="Follow melody", zh="跟随旋律"),
    control_score=dict(en="Follow score", zh="跟随乐谱"),
    examples_label=dict(en="Load an example", zh="载入示例"),
    example_choice_0=dict(en="—", zh="—"),
    example_choice_1=dict(en="Example 1: Mandarin → Mandarin (melody), Start singing synthesis!", zh="样例 1: 普通话 → 普通话 (melody), 开始歌声合成吧!"),
    example_choice_2=dict(en="Example 2: Mandarin → Cantonese (melody), Start singing synthesis!", zh="样例 2: 普通话 → 粤语 (melody), 开始歌声合成吧!"),
    example_choice_3=dict(en="Example 3: Mandarin → English (melody), Start singing synthesis!", zh="样例 3: 普通话 → 英语 (melody), 开始歌声合成吧!"),
    example_choice_4=dict(en="Example 4: Mandarin → Music (score), Start singing synthesis!", zh="样例 4: 普通话 → 音乐 (score), 开始歌声合成吧!"),
    status_waiting=dict(en="Waiting for audio", zh="等待音频"),
    status_processing=dict(en="Transcribing audio", zh="正在转录音频"),
    status_ready=dict(en="Ready to generate", zh="已准备完成"),
    status_failed=dict(en="Processing failed", zh="处理失败"),
    result_waiting=dict(en="Your generated performance will appear here.", zh="生成完成后，可在这里试听结果。"),
    result_processing=dict(en="Generating. CPU inference may take a while.", zh="正在生成，CPU 推理可能需要一些时间。"),
    result_ready=dict(en="Generation complete", zh="生成完成"),
    result_failed=dict(en="Generation failed. Check the application log and try again.", zh="生成失败，请查看应用日志后重试。"),
)

_GLOBAL_LANG: Literal["zh", "en"] = "zh"


def _i18n(key: str) -> str:
    return _I18N_KEY2LANG[key][_GLOBAL_LANG]


def _load_example(choice_value):
    """Load an example and mark its bundled metadata as ready."""
    output_count = 18
    if choice_value is None:
        return [gr.update()] * output_count + [0]

    choice_to_index = {
        _i18n("example_choice_1"): 1,
        _i18n("example_choice_2"): 2,
        _i18n("example_choice_3"): 3,
        _i18n("example_choice_4"): 4,
    }

    idx = 0
    if isinstance(choice_value, int):
        idx = 0 if choice_value <= 0 else min(choice_value - 1, len(EXAMPLES_LIST) - 1)
    else:
        idx = choice_to_index.get(choice_value, 0)

    if idx <= 0:
        return [gr.update()] * output_count + [0]

    list_idx = idx - 1
    if list_idx >= len(EXAMPLES_LIST):
        return [gr.update()] * output_count + [0]

    row = EXAMPLES_LIST[list_idx]
    ready = _status_payload("ready")
    return [
        row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10],
        ready, _status_markup(ready), ready, _status_markup(ready),
        gr.update(visible=False, interactive=False), gr.update(visible=False, interactive=False), gr.update(interactive=True),
        2,
    ]


def _status_payload(kind: str, detail: str | None = None) -> dict:
    return {"kind": kind, "detail": detail}


def _status_markup(payload: dict | str | None) -> str:
    if isinstance(payload, str):
        payload = _status_payload(payload)
    payload = payload or _status_payload("waiting")
    kind = payload.get("kind", "waiting")
    label = _i18n(f"status_{kind}")
    detail = payload.get("detail")
    detail_html = f'<span class="status-detail">{html.escape(str(detail)[:220])}</span>' if detail else ""
    return f'<div class="process-status status-{kind}"><span class="status-dot"></span><strong>{label}</strong>{detail_html}</div>'


def _result_payload(kind: str) -> dict:
    return {"kind": kind}


def _result_markup(payload: dict | str | None) -> str:
    if isinstance(payload, str):
        payload = _result_payload(payload)
    payload = payload or _result_payload("waiting")
    kind = payload.get("kind", "waiting")
    return f'<div class="result-status result-{kind}">{html.escape(_i18n(f"result_{kind}"))}</div>'


def _mark_processing():
    payload = _status_payload("processing")
    return payload, _status_markup(payload), gr.update(visible=False, interactive=False), gr.update(interactive=False)


def _mark_result_processing():
    payload = _result_payload("processing")
    return payload, _result_markup(payload), gr.update(interactive=False)


def _update_generate_button(prompt_metadata, target_metadata):
    ready = bool(_resolve_file_path(prompt_metadata) and _resolve_file_path(target_metadata))
    return gr.update(interactive=ready)


def _clear_prompt_state(_audio, skip_count):
    if skip_count and skip_count > 0:
        ready = _status_payload("ready")
        return gr.skip(), max(0, skip_count - 1), ready, _status_markup(ready), gr.update(visible=False, interactive=False), gr.skip()
    waiting = _status_payload("waiting")
    return None, 0, waiting, _status_markup(waiting), gr.update(visible=False, interactive=False), gr.update(interactive=False)


def _clear_target_state(_audio, skip_count):
    if skip_count and skip_count > 0:
        ready = _status_payload("ready")
        return gr.skip(), max(0, skip_count - 1), ready, _status_markup(ready), gr.update(visible=False, interactive=False), gr.skip()
    waiting = _status_payload("waiting")
    return None, 0, waiting, _status_markup(waiting), gr.update(visible=False, interactive=False), gr.update(interactive=False)


def _get_device() -> str:
    """Use CUDA if available, else CPU (e.g. for CI or CPU-only environments)."""
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _session_dir() -> Path:
    # Use per-call timestamped session dir to avoid cross-request collisions.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return ROOT / "outputs" / "gradio" / timestamp


def _print_exception(context: str) -> None:
    print(f"[{context}]\n{traceback.format_exc()}", file=sys.stderr, flush=True)


def _get_lyric_lang_choices():
    """Lyric language dropdown (display, value) for current UI language."""
    return [
        (_i18n("lyric_lang_mandarin"), "Mandarin"),
        (_i18n("lyric_lang_cantonese"), "Cantonese"),
        (_i18n("lyric_lang_english"), "English"),
    ]


def _resolve_file_path(x):
    """Gradio file input can be path string or (path, None) tuple."""
    if x is None:
        return None
    if isinstance(x, tuple):
        x = x[0]
    return x if (x and os.path.isfile(x)) else None


def _normalize_audio_input(audio):
    """Normalize Gradio audio input to a filepath string."""
    return audio[0] if isinstance(audio, tuple) else audio


def _trim_and_save_audio(src_audio_path: str, dst_wav_path: Path, max_sec: int, sr: int = SAMPLE_RATE) -> None:
    """Load audio as mono, trim to max_sec, and save as wav for preprocess."""
    audio_data, _ = librosa.load(src_audio_path, sr=sr, mono=True)
    audio_data = audio_data[: max_sec * sr]
    sf.write(dst_wav_path, audio_data, sr)


def _yes_no_to_bool(value, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() == "yes"


def _control_to_internal(control: str) -> str:
    if control in ("melody", "melody-controlled"):
        return "melody"
    return "score"


class AppState:
    def __init__(self, use_fp16: bool = False) -> None:
        self.device = _get_device()
        self.use_fp16 = use_fp16 and ("cuda" in self.device)
        self.preprocess_pipeline = PreprocessPipeline(
            device=self.device,
            language="Mandarin",
            save_dir=str(ROOT / "outputs" / "gradio" / "_placeholder" / "transcriptions"),
            vocal_sep=True,
            max_merge_duration=60000,
        )
        config = load_config("soulxsinger/config/soulxsinger.yaml")
        self.svs_config = config
        self.svs_model = build_svs_model(
            model_path="pretrained_models/SoulX-Singer/model.pt",
            config=config,
            device=self.device,
            use_fp16=self.use_fp16,
        )
        self.phoneset_path = "soulxsinger/utils/phoneme/phone_set.json"

    def run_preprocess(
        self,
        audio_path: Path,
        save_path: Path,
        vocal_sep: bool,
        lyric_lang: str,
        max_merge_duration: int
    ) -> Tuple[bool, str]:
        try:
            self.preprocess_pipeline.save_dir = str(save_path)
            self.preprocess_pipeline.run(
                audio_path=str(audio_path),
                vocal_sep=vocal_sep,
                max_merge_duration=max_merge_duration,
                language=lyric_lang or "Mandarin",
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True, f"preprocess {audio_path} done"
        except Exception as e:
            return False, f"preprocess failed: {e}"

    def run_svs(
        self,
        control: str,
        session_base: Path,
        auto_shift: bool,
        pitch_shift: int,
    ) -> Tuple[bool, str, Path | None, Path | None, Path | None]:
        if control not in ("melody", "score"):
            control = "score"
        save_dir = session_base / "generated"
        save_dir.mkdir(parents=True, exist_ok=True)
        class Args:
            pass
        args = Args()
        args.device = self.device
        args.model_path = "pretrained_models/SoulX-Singer/model.pt"
        args.config = "soulxsinger/config/soulxsinger.yaml"
        args.prompt_wav_path = str(session_base / "audio" / "prompt.wav")
        prompt_meta_path = session_base / "transcriptions" / "prompt" / "metadata.json"
        target_meta_path = session_base / "transcriptions" / "target" / "metadata.json"
        args.prompt_metadata_path = str(prompt_meta_path)
        args.target_metadata_path = str(target_meta_path)
        args.phoneset_path = self.phoneset_path
        args.save_dir = str(save_dir)
        args.auto_shift = auto_shift
        args.pitch_shift = int(pitch_shift)
        args.control = control
        args.use_fp16 = self.use_fp16
        try:
            svs_process(args, self.svs_config, self.svs_model)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            generated = save_dir / "generated.wav"
            if not generated.exists():
                return False, f"inference finished but {generated} not found", None, prompt_meta_path, target_meta_path
            return True, "svs inference done", generated, prompt_meta_path, target_meta_path
        except Exception as e:
            return False, f"svs inference failed: {e}", None, prompt_meta_path, target_meta_path

    def run_svs_from_paths(
        self,
        prompt_wav_path: str,
        prompt_metadata_path: str,
        target_metadata_path: str,
        control: str,
        auto_shift: bool,
        pitch_shift: int,
        save_dir: Path | None = None,
    ) -> Tuple[bool, str, Path | None]:
        """Run SVS from explicit prompt wav and metadata paths."""
        if save_dir is None:
            import uuid
            save_dir = ROOT / "outputs" / "gradio" / "synthesis" / str(uuid.uuid4())[:8]
        save_dir = Path(save_dir)
        audio_dir = save_dir / "audio"
        prompt_meta_dir = save_dir / "transcriptions" / "prompt"
        target_meta_dir = save_dir / "transcriptions" / "target"
        audio_dir.mkdir(parents=True, exist_ok=True)
        prompt_meta_dir.mkdir(parents=True, exist_ok=True)
        target_meta_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(prompt_wav_path, audio_dir / "prompt.wav")
        shutil.copy2(prompt_metadata_path, prompt_meta_dir / "metadata.json")
        shutil.copy2(target_metadata_path, target_meta_dir / "metadata.json")
        ok, msg, merged, _, _ = self.run_svs(
            control=control,
            session_base=save_dir,
            auto_shift=auto_shift,
            pitch_shift=pitch_shift,
        )
        if not ok or merged is None:
            return False, msg or "svs failed", None
        return True, "svs inference done", merged


APP_STATE = AppState(use_fp16="--fp16" in sys.argv)


def _transcribe_prompt(
    prompt_audio,
    prompt_lyric_lang: str,
    prompt_vocal_sep,
    prompt_max_sec: int = PROMPT_MAX_SEC_DEFAULT,
):
    try:
        prompt_audio = _normalize_audio_input(prompt_audio)
        if prompt_audio is None:
            waiting = _status_payload("waiting")
            return None, waiting, _status_markup(waiting), gr.update(visible=False, interactive=False)

        session_base = _session_dir()
        prompt_meta_path = session_base / "transcriptions" / "prompt" / "metadata.json"
        audio_dir = session_base / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        transfer_prompt_path = audio_dir / "prompt.wav"
        _trim_and_save_audio(prompt_audio, transfer_prompt_path, prompt_max_sec)

        prompt_ok, prompt_msg = APP_STATE.run_preprocess(
            audio_path=transfer_prompt_path,
            save_path=session_base / "transcriptions" / "prompt",
            vocal_sep=_yes_no_to_bool(prompt_vocal_sep, default=False),
            lyric_lang=prompt_lyric_lang or "Mandarin",
            max_merge_duration=PROMPT_MAX_MERGE_DURATION_MS,
        )
        if not prompt_ok or not prompt_meta_path.exists():
            print(prompt_msg, file=sys.stderr, flush=True)
            failed = _status_payload("failed", prompt_msg)
            return None, failed, _status_markup(failed), gr.update(visible=True, interactive=True)

        ready = _status_payload("ready")
        return str(prompt_meta_path), ready, _status_markup(ready), gr.update(visible=False, interactive=False)
    except Exception as exc:
        _print_exception("_transcribe_prompt")
        failed = _status_payload("failed", str(exc))
        return None, failed, _status_markup(failed), gr.update(visible=True, interactive=True)


def _transcribe_target(
    target_audio,
    target_lyric_lang: str,
    target_vocal_sep,
    target_max_sec: int = TARGET_MAX_SEC_DEFAULT,
):
    try:
        target_audio = _normalize_audio_input(target_audio)
        if target_audio is None:
            waiting = _status_payload("waiting")
            return None, None, waiting, _status_markup(waiting), gr.update(visible=False, interactive=False)

        session_base = _session_dir()
        target_meta_path = session_base / "transcriptions" / "target" / "metadata.json"
        target_vocal_path = session_base / "transcriptions" / "target" / "vocal.wav"
        audio_dir = session_base / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        transfer_target_path = audio_dir / "target.wav"
        _trim_and_save_audio(target_audio, transfer_target_path, target_max_sec)

        target_ok, target_msg = APP_STATE.run_preprocess(
            audio_path=transfer_target_path,
            save_path=session_base / "transcriptions" / "target",
            vocal_sep=_yes_no_to_bool(target_vocal_sep, default=True),
            lyric_lang=target_lyric_lang or "Mandarin",
            max_merge_duration=TARGET_MAX_MERGE_DURATION_MS,
        )
        if not target_ok or not target_meta_path.exists():
            print(target_msg, file=sys.stderr, flush=True)
            failed = _status_payload("failed", target_msg)
            return None, None, failed, _status_markup(failed), gr.update(visible=True, interactive=True)

        ready = _status_payload("ready")
        target_vocal_file = str(target_vocal_path) if target_vocal_path.exists() else None
        return str(target_meta_path), target_vocal_file, ready, _status_markup(ready), gr.update(visible=False, interactive=False)
    except Exception as exc:
        _print_exception("_transcribe_target")
        failed = _status_payload("failed", str(exc))
        return None, None, failed, _status_markup(failed), gr.update(visible=True, interactive=True)


def _run_synthesis(
    prompt_audio,
    prompt_metadata,
    target_metadata,
    control: str,
    auto_shift,
    pitch_shift,
    seed: int,
):
    """Run singing synthesis from prompt audio + prompt metadata + target metadata."""
    try:
        prompt_audio = _normalize_audio_input(prompt_audio)
        prompt_wav_path = prompt_audio
        prompt_meta_path = _resolve_file_path(prompt_metadata)
        target_meta_path = _resolve_file_path(target_metadata)
        if not prompt_wav_path or not os.path.isfile(prompt_wav_path):
            gr.Warning(message=_i18n("warn_missing_synthesis"))
            failed = _result_payload("failed")
            return None, failed, _result_markup(failed), gr.update(interactive=True)
        if not prompt_meta_path or not os.path.isfile(prompt_meta_path):
            gr.Warning(message=_i18n("warn_missing_synthesis"))
            failed = _result_payload("failed")
            return None, failed, _result_markup(failed), gr.update(interactive=True)
        if not target_meta_path or not os.path.isfile(target_meta_path):
            gr.Warning(message=_i18n("warn_missing_synthesis"))
            failed = _result_payload("failed")
            return None, failed, _result_markup(failed), gr.update(interactive=True)
        control = _control_to_internal(control)
        auto_shift = _yes_no_to_bool(auto_shift, default=True)
        seed = int(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        ok, msg, merged = APP_STATE.run_svs_from_paths(
            prompt_wav_path=prompt_wav_path,
            prompt_metadata_path=prompt_meta_path,
            target_metadata_path=target_meta_path,
            control=control,
            auto_shift=auto_shift,
            pitch_shift=int(pitch_shift),
        )
        if not ok or merged is None:
            print(msg or "synthesis failed", file=sys.stderr, flush=True)
            failed = _result_payload("failed")
            return None, failed, _result_markup(failed), gr.update(interactive=True)
        ready = _result_payload("ready")
        return str(merged), ready, _result_markup(ready), gr.update(interactive=True)
    except Exception:
        _print_exception("_run_synthesis")
        failed = _result_payload("failed")
        return None, failed, _result_markup(failed), gr.update(interactive=True)


APP_CSS = """
:root {
  --studio-bg: #f2f5f6;
  --studio-surface: #ffffff;
  --studio-ink: #17252d;
  --studio-muted: #5c6b73;
  --studio-line: #d8e0e3;
  --studio-accent: #008e93;
  --studio-accent-dark: #006f73;
  --studio-success: #16784b;
  --studio-warning: #9a6500;
  --studio-danger: #b33a3a;
}
html, body, .gradio-container { background: var(--studio-bg) !important; }
.gradio-container {
  color: var(--studio-ink) !important;
  font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", sans-serif !important;
  max-width: none !important;
  padding: 0 !important;
}
#studio-app .main { padding: 0 !important; }
#studio-shell { max-width: 1180px; margin: 34px auto 40px; padding: 0 28px; box-sizing: border-box; gap: 18px; width: 100%; }
.topbar {
  align-items: center !important;
  border-bottom: 1px solid var(--studio-line);
  margin-bottom: 26px !important;
  padding: 16px 18px 18px;
  background: rgba(255,255,255,.88);
  border-radius: 14px;
  box-shadow: 0 12px 30px rgba(23,37,45,.06);
  gap: 16px !important;
}
.brand-lockup { display: flex; align-items: center; gap: 12px; min-width: 300px; min-height: 48px; }
.brand-mark {
  display: flex; align-items: end; justify-content: center; gap: 3px;
  width: 42px; height: 42px; padding: 10px; box-sizing: border-box;
  border-radius: 8px; background: #102b32;
}
.brand-mark span { display: block; width: 4px; border-radius: 2px 2px 0 0; background: #27c4c3; }
.brand-mark span:nth-child(1) { height: 10px; }
.brand-mark span:nth-child(2) { height: 20px; }
.brand-mark span:nth-child(3) { height: 15px; }
.brand-mark span { transform-origin: bottom center; }
.brand-mark span:nth-child(2) { animation-delay: .18s; }
.brand-mark span:nth-child(3) { animation-delay: .34s; }
@keyframes signal-rise { 0%, 100% { transform: scaleY(.72); opacity: .72; } 50% { transform: scaleY(1.12); opacity: 1; } }
.brand-name { font-size: 20px; line-height: 1.1; font-weight: 720; letter-spacing: 0; color: var(--studio-ink); white-space: nowrap; }
.brand-subtitle { margin-top: 4px; color: var(--studio-muted); font-size: 13px; }
.model-ready { margin-left: auto; display: inline-flex; align-items: center; gap: 7px; color: var(--studio-success); font-size: 13px; font-weight: 650; white-space: nowrap; }
.model-ready::before { content: ""; width: 8px; height: 8px; border-radius: 50%; background: var(--studio-success); }
.studio-intro { display:flex; align-items:center; justify-content:space-between; gap:28px; margin: 0 0 30px; padding: 24px 28px; border-radius: 16px; color:#efffff; background:#102b32; box-shadow: 0 20px 44px rgba(16,43,50,.18); overflow:hidden; position:relative; }

.intro-copy { position:relative; z-index:1; }

.intro-title { font-size:27px; font-weight:760; line-height:1.1; letter-spacing:0; margin:0 0 8px; color:#efffff !important; }
.intro-desc { margin:0; color:#b8d3d4; font-size:13px; max-width:50ch; }
.intro-visual { position:relative; z-index:1; display:flex; align-items:center; gap:5px; height:58px; padding:0 4px; }
.signal-bar { width:4px; min-height:10px; border-radius:3px; background:#27c4c3; transform-origin:center; opacity:.9; }
.workflow-steps { display:flex; gap:8px; margin: 0 0 22px; overflow:auto; }
.workflow-step { display:flex; align-items:center; gap:8px; white-space:nowrap; padding:9px 12px; border:1px solid var(--studio-line); border-radius:10px; background:#fff; color:var(--studio-muted); font-size:12px; font-weight:650; }
.workflow-step.active { color:var(--studio-ink); border-color:#8ad8d7; background:#f0fbfb; }
.step-index { width:20px; height:20px; border-radius:50%; display:grid; place-items:center; background:#e8eff0; color:#567078; font-size:11px; }
.workflow-step.active .step-index { background:var(--studio-accent); color:#fff; }
.section-heading { margin: 0 0 14px !important; }
.section-heading h2 { margin: 0 0 4px !important; font-size: 19px !important; letter-spacing: 0 !important; }
.section-heading h3 { margin: 0 0 3px !important; font-size: 16px !important; letter-spacing: 0 !important; }
.section-heading p { max-width: 72ch; margin: 0 !important; color: var(--studio-muted) !important; font-size: 13px; }
.sources-grid { gap: 18px !important; margin-bottom: 28px !important; }
.source-panel {
  min-width: 380px !important; padding: 18px !important;
  border: 1px solid var(--studio-line); border-radius: 14px;
  background: var(--studio-surface);
  box-shadow: 0 1px 2px rgba(23, 37, 45, .05), 0 10px 28px rgba(23, 37, 45, .035);
  position: relative; overflow: hidden;
}

.source-panel .audio-container { min-height: 156px; }
.source-controls { gap: 12px !important; align-items: end !important; }
.process-status {
  min-height: 34px; display: flex; align-items: center; gap: 8px;
  margin-top: 8px; padding: 7px 10px; border-radius: 6px;
  background: #f5f7f8; color: var(--studio-muted); font-size: 12px;
}
.process-status strong { font-weight: 680; color: inherit; }
.status-dot { width: 8px; height: 8px; border-radius: 50%; background: #87969d; flex: none; }
.status-detail { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.status-processing { color: var(--studio-warning); background: #fff8e7; }
.status-processing .status-dot { background: var(--studio-warning); animation: status-pulse 1.1s ease-in-out infinite alternate; }
.status-ready { color: var(--studio-success); background: #edf8f1; }
.status-ready .status-dot { background: var(--studio-success); }
.status-failed { color: var(--studio-danger); background: #fff0f0; }
.status-failed .status-dot { background: var(--studio-danger); }
@keyframes status-pulse { to { opacity: .35; } }
.settings-band, .result-band {
  padding: 18px; border-top: 1px solid var(--studio-line); background: var(--studio-surface);
}
.settings-band { border-bottom: 1px solid var(--studio-line); }
.settings-row { gap: 14px !important; align-items: end !important; }
.primary-action { min-width: 220px !important; height: 42px !important; }
.primary-action button, button.primary {
  background: var(--studio-accent) !important; border-color: var(--studio-accent) !important;
  color: white !important; font-weight: 700 !important;
}
.primary-action button:hover, button.primary:hover { background: var(--studio-accent-dark) !important; }
.primary-action button:disabled, button.primary:disabled { background: #aab7bb !important; border-color: #aab7bb !important; color: #f8fafb !important; }
.retry-action { margin-top: 8px !important; max-width: 150px; }
.result-band { margin-top: 26px !important; border-radius: 14px; border: 1px solid var(--studio-line); box-shadow: 0 18px 42px rgba(23,37,45,.06); }
.result-status { margin: 2px 0 12px; color: var(--studio-muted); font-size: 13px; }
.result-processing { color: var(--studio-warning); }
.result-ready { color: var(--studio-success); font-weight: 650; }
.result-failed { color: var(--studio-danger); font-weight: 650; }
#lang_choice_radio { min-width: 205px; }
#lang_choice_radio .wrap { padding: 0 !important; }
footer, .built-with, button[aria-label*="API"], button[aria-label*="设置"], button[aria-label*="Settings"] { display: none !important; }
button, input, textarea, select { letter-spacing: 0 !important; }
*:focus-visible { outline: 2px solid #00aeb3 !important; outline-offset: 2px !important; }
::selection { background: #bcebed; color: #102b32; }
@media (hover: hover) {
  .source-panel:hover { border-color: #8dbfbd; }
}
@media (max-width: 1050px) {
  #studio-shell { margin-top: 30px; padding: 0 22px; }
  .source-panel { min-width: 320px !important; }
  .brand-lockup { min-width: 270px; }
  .model-ready { margin-left: 2px; font-size: 0; }
  .studio-intro { align-items:flex-start; padding:20px; }
  .intro-visual { display:none; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation-duration: 0.01ms !important; animation-iteration-count: 1 !important; transition-duration: 0.01ms !important; }
}
#workflow { border: 0; }
#workflow > .tab-nav { gap: 8px; border: 0; margin-bottom: 24px; padding: 0; }
#workflow > .tab-nav button { border: 0; border-radius: 8px; padding: 10px 22px; font-size: 14px; color: #51636a; background: #e5ebed; }
#workflow > .tab-nav button.selected { color: white; background: #153b42; }
#workflow > .tabitem { padding: 0; border: 0; background: transparent; }
.source-panel { padding: 22px !important; box-shadow: none; }
.source-panel .audio-container { min-height: 160px; }
.source-panel .section-heading { margin-bottom: 12px !important; }
.sources-grid { margin-bottom: 4px !important; }
.section-heading h2 { font-size: 22px !important; }
.source-controls { background: transparent !important; }
.settings-band { padding: 28px; border: 1px solid var(--studio-line); border-radius: 14px; }
.settings-row { flex-wrap: wrap; }
.settings-row > * { min-width: 210px !important; }
.result-band { margin-top: 0 !important; padding: 28px; box-shadow: none; }
.studio-intro { margin: 0; box-shadow: none; min-height: 110px; box-sizing: border-box; }
.topbar { background: transparent; box-shadow: none; padding: 0 0 18px; border-radius: 0; margin-bottom: 0 !important; }
#studio-shell input { caret-color: #007e80; }
#studio-shell input[type=number] { font-variant-numeric: tabular-nums; }
html { scrollbar-color: #a6babd #f2f5f6; }
.studio-art { width: 210px; height: 76px; object-fit: contain; }
@media(max-width: 1050px) { .studio-art { width: 160px; } .brand-lockup { min-width: 230px; } }

"""

APP_MOTION = r"""
(() => {
  if (window.__soulxMotionInstalled || !window.gsap) return;
  window.__soulxMotionInstalled = true;
  const { gsap } = window;
  const preference = matchMedia('(prefers-reduced-motion: reduce)');
  let pressed = null;
  const reset = () => {
    if (pressed) gsap.to(pressed, {scale: 1, duration: .15, overwrite: true, clearProps: 'transform'});
    pressed = null;
  };
  document.addEventListener('pointerdown', e => {
    if (preference.matches) return;
    pressed = e.target.closest('button:not(:disabled)');
    if (pressed) gsap.to(pressed, {scale: .98, duration: .1, overwrite: true});
  });
  ['pointerup','pointercancel','blur'].forEach(name => window.addEventListener(name, reset));
  document.addEventListener('click', e => {
    if (preference.matches || !e.target.closest('#workflow [role=tab]')) return;
    requestAnimationFrame(() => {
      document.querySelectorAll('#workflow [role=tabpanel]').forEach(panel => {
        if (panel.getBoundingClientRect().height) gsap.fromTo(panel, {x: 8}, {x: 0, duration: .24, ease: 'power3.out', overwrite: true, clearProps: 'transform'});
      });
    });
  });
  document.addEventListener('dragenter', e => {
    const zone = e.target.closest('.audio-container');
    if (zone && !preference.matches) gsap.to(zone, {scale: 1.008, duration: .15, overwrite: true});
  });
  ['drop','dragend','dragleave'].forEach(name => document.addEventListener(name, () => {
    gsap.to('.audio-container', {scale: 1, duration: .15, overwrite: true, clearProps: 'transform'});
  }));
  // Animate only processing feedback; never hide Gradio-owned elements.
  const seen = new WeakMap();
  let scheduled = false;
  const sync = () => {
    scheduled = false;
    document.querySelectorAll('.process-status, .result-status').forEach(node => {
      const state = node.className + node.textContent;
      if (seen.get(node) === state) return;
      seen.set(node, state);
      if (!preference.matches && node.getBoundingClientRect().height) {
        gsap.fromTo(node, {x: 4}, {x: 0, duration: .2, overwrite: true, clearProps: 'transform'});
      }
    });
  };
  new MutationObserver(() => {
    if (!scheduled) { scheduled = true; requestAnimationFrame(sync); }
  }).observe(document.documentElement, {childList:true, subtree:true, characterData:true});
})();
"""

APP_HEAD = f"<script>{GSAP_SOURCE}</script><script>{APP_MOTION}</script>"


def _intro_markup() -> str:
    title = "让音色与旋律相遇" if _GLOBAL_LANG == "zh" else "Your voice. A new performance."
    desc = "上传两段音频，自动转录后生成新的歌声。" if _GLOBAL_LANG == "zh" else "Add a reference voice and a song. Transcription is automatic."
    return f'<div class="studio-intro"><div class="intro-copy"><h1 class="intro-title">{title}</h1><p class="intro-desc">{desc}</p></div></div>'


def _brand_markup() -> str:
    return (
        '<div class="brand-lockup">'
        '<div class="brand-mark" aria-hidden="true"><span></span><span></span><span></span></div>'
        '<div><div class="brand-name">SoulX-Singer</div><div class="brand-subtitle">Singing Voice Studio</div></div>'
        f'<div class="model-ready" title="{html.escape(_i18n("service_ready"))}">{html.escape(_i18n("service_ready"))}</div>'
        '</div>'
    )


def _heading(level: int, title_key: str, desc_key: str | None = None) -> str:
    description = f"\n\n{_i18n(desc_key)}" if desc_key else ""
    return f'{"#" * level} {_i18n(title_key)}{description}'


def render_interface() -> gr.Blocks:
    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.cyan,
        neutral_hue=gr.themes.colors.slate,
        radius_size=gr.themes.sizes.radius_sm,
    )
    with gr.Blocks(
        title="SoulX-Singer",
        theme=theme,
        css=APP_CSS,
        head=APP_HEAD,
        fill_height=False,
        elem_id="studio-app",
    ) as page:
        skip_clear_metadata_count = gr.State(0)
        prompt_metadata = gr.State(None)
        target_metadata = gr.State(None)
        target_vocal = gr.State(None)
        prompt_status_state = gr.State(_status_payload("waiting"))
        target_status_state = gr.State(_status_payload("waiting"))
        result_status_state = gr.State(_result_payload("waiting"))

        with gr.Column(elem_id="studio-shell"):
            with gr.Row(elem_classes=["topbar"]):
                brand_header = gr.HTML(_brand_markup(), scale=3)
                _example_choices = [_i18n("example_choice_0"), _i18n("example_choice_1"), _i18n("example_choice_2"), _i18n("example_choice_3"), _i18n("example_choice_4")]
                example_choice = gr.Dropdown(
                    label=_i18n("examples_label"), choices=_example_choices,
                    value=_example_choices[0], interactive=True, scale=2,
                )
                lang_choice = gr.Radio(
                    choices=["中文", "English"], value="中文", label=_i18n("display_lang_label"),
                    type="index", interactive=True, elem_id="lang_choice_radio", scale=1,
                )

            intro_header = gr.HTML(_intro_markup())
            with gr.Tabs(elem_id="workflow", selected="sources") as workflow:
                with gr.Tab("声音素材", id="sources") as sources_tab:
                    sources_heading = gr.Markdown(_heading(2, "sources_title", "sources_desc"), elem_classes=["section-heading"])
                    with gr.Row(equal_height=True, elem_classes=["sources-grid"]):
                        with gr.Column(scale=1, elem_classes=["source-panel"]):
                            prompt_heading = gr.Markdown(_heading(3, "prompt_title", "prompt_desc"), elem_classes=["section-heading"])
                            with gr.Row(elem_classes=["source-controls"]):
                                prompt_lyric_lang = gr.Dropdown(
                                    label=_i18n("prompt_lyric_lang_label"), choices=_get_lyric_lang_choices(),
                                    value="Mandarin", interactive=True, scale=2,
                                )
                                prompt_vocal_sep = gr.Checkbox(
                                    label=_i18n("prompt_vocal_sep_label"), value=False, interactive=True, scale=1,
                                )
                            prompt_audio = gr.Audio(
                                label=_i18n("prompt_audio_label"), type="filepath", editable=False,
                                interactive=True, elem_classes=["audio-container"],
                            )
                            prompt_status = gr.HTML(_status_markup("waiting"))
                            prompt_retry = gr.Button(
                                value=_i18n("retry_label"), variant="secondary", size="sm",
                                visible=False, interactive=False, elem_classes=["retry-action"],
                            )

                        with gr.Column(scale=1, elem_classes=["source-panel"]):
                            target_heading = gr.Markdown(_heading(3, "target_title", "target_desc"), elem_classes=["section-heading"])
                            with gr.Row(elem_classes=["source-controls"]):
                                target_lyric_lang = gr.Dropdown(
                                    label=_i18n("target_lyric_lang_label"), choices=_get_lyric_lang_choices(),
                                    value="Mandarin", interactive=True, scale=2,
                                )
                                target_vocal_sep = gr.Checkbox(
                                    label=_i18n("target_vocal_sep_label"), value=True, interactive=True, scale=1,
                                )
                            target_audio = gr.Audio(
                                label=_i18n("target_audio_label"), type="filepath", editable=False,
                                interactive=True, elem_classes=["audio-container"],
                            )
                            target_status = gr.HTML(_status_markup("waiting"))
                            target_retry = gr.Button(
                                value=_i18n("retry_label"), variant="secondary", size="sm",
                                visible=False, interactive=False, elem_classes=["retry-action"],
                            )

                    next_settings = gr.Button("调整生成设置 →", variant="secondary")
                with gr.Tab("生成设置", id="settings") as settings_tab:
                    with gr.Column(elem_classes=["settings-band"]):
                        settings_heading = gr.Markdown(_heading(2, "settings_title"), elem_classes=["section-heading"])
                        with gr.Row(equal_height=True, elem_classes=["settings-row"]):
                            control_radio = gr.Radio(
                                choices=[(_i18n("control_melody"), "melody-controlled"), (_i18n("control_score"), "score-controlled")],
                                value="score-controlled", label=_i18n("control_type_label"), scale=2,
                            )
                            auto_shift = gr.Checkbox(label=_i18n("auto_shift_label"), value=True, interactive=True, scale=1)
                            pitch_shift = gr.Number(
                                label=_i18n("pitch_shift_label"), value=0, minimum=-36, maximum=36,
                                step=1, interactive=True, scale=1,
                            )
                            seed_input = gr.Number(label=_i18n("seed_label"), value=12306, step=1, interactive=True, scale=1)
                            synthesis_btn = gr.Button(
                                value=_i18n("synthesis_btn_label"), variant="primary", size="lg",
                                interactive=False, scale=2, elem_classes=["primary-action"],
                            )

                with gr.Tab("试听结果", id="result") as result_tab:
                    with gr.Column(elem_classes=["result-band"]):
                        result_heading = gr.Markdown(_heading(2, "result_title"), elem_classes=["section-heading"])
                        result_status = gr.HTML(_result_markup("waiting"))
                        output_audio = gr.Audio(
                            label=_i18n("generated_audio_label"), type="filepath", interactive=False,
                        )

            next_settings.click(lambda: gr.update(selected="settings"), outputs=workflow, queue=False)
            synthesis_btn.click(lambda: gr.update(selected="result"), outputs=workflow, queue=False)

            example_choice.change(
                fn=_load_example,
                inputs=[example_choice],
                outputs=[
                    prompt_audio, target_audio, prompt_metadata, target_metadata,
                    prompt_lyric_lang, target_lyric_lang, control_radio,
                    prompt_vocal_sep, target_vocal_sep, auto_shift, pitch_shift,
                    prompt_status_state, prompt_status, target_status_state, target_status,
                    prompt_retry, target_retry, synthesis_btn, skip_clear_metadata_count,
                ],
            )

            def _change_component_language(lang, prompt_state, target_state, result_state):
                global _GLOBAL_LANG
                _GLOBAL_LANG = ["zh", "en"][lang]
                control_choices = [(_i18n("control_melody"), "melody-controlled"), (_i18n("control_score"), "score-controlled")]
                return [
                    _brand_markup(), _intro_markup(),
                    gr.update(label="声音素材" if lang == 0 else "Audio sources"),
                    gr.update(label="生成设置" if lang == 0 else "Generation"),
                    gr.update(label="试听结果" if lang == 0 else "Listen"),
                    gr.update(value="调整生成设置 →" if lang == 0 else "Generation settings →"),
                    gr.update(value=_heading(2, "sources_title", "sources_desc")),
                    gr.update(value=_heading(3, "prompt_title", "prompt_desc")),
                    gr.update(value=_heading(3, "target_title", "target_desc")),
                    gr.update(value=_heading(2, "settings_title")),
                    gr.update(value=_heading(2, "result_title")),
                    gr.update(label=_i18n("prompt_audio_label")),
                    gr.update(label=_i18n("target_audio_label")),
                    gr.update(label=_i18n("prompt_lyric_lang_label"), choices=_get_lyric_lang_choices()),
                    gr.update(label=_i18n("target_lyric_lang_label"), choices=_get_lyric_lang_choices()),
                    gr.update(label=_i18n("prompt_vocal_sep_label")),
                    gr.update(label=_i18n("target_vocal_sep_label")),
                    gr.update(label=_i18n("control_type_label"), choices=control_choices),
                    gr.update(label=_i18n("auto_shift_label")),
                    gr.update(label=_i18n("pitch_shift_label")),
                    gr.update(label=_i18n("seed_label")),
                    gr.update(value=_i18n("synthesis_btn_label")),
                    gr.update(label=_i18n("generated_audio_label")),
                    gr.update(label=_i18n("display_lang_label")),
                    gr.update(label=_i18n("examples_label"), choices=[_i18n("example_choice_0"), _i18n("example_choice_1"), _i18n("example_choice_2"), _i18n("example_choice_3"), _i18n("example_choice_4")], value=_i18n("example_choice_0")),
                    gr.update(value=_i18n("retry_label")),
                    gr.update(value=_i18n("retry_label")),
                    _status_markup(prompt_state), _status_markup(target_state), _result_markup(result_state),
                ]

            lang_choice.change(
                fn=_change_component_language,
                inputs=[lang_choice, prompt_status_state, target_status_state, result_status_state],
                outputs=[
                    brand_header, intro_header, sources_tab, settings_tab, result_tab, next_settings, sources_heading, prompt_heading, target_heading, settings_heading, result_heading,
                    prompt_audio, target_audio, prompt_lyric_lang, target_lyric_lang,
                    prompt_vocal_sep, target_vocal_sep, control_radio, auto_shift,
                    pitch_shift, seed_input, synthesis_btn, output_audio, lang_choice, example_choice,
                    prompt_retry, target_retry, prompt_status, target_status, result_status,
                ],
            )

            prompt_audio.change(
                fn=_clear_prompt_state,
                inputs=[prompt_audio, skip_clear_metadata_count],
                outputs=[prompt_metadata, skip_clear_metadata_count, prompt_status_state, prompt_status, prompt_retry, synthesis_btn],
            )
            prompt_upload = prompt_audio.upload(
                fn=_mark_processing,
                outputs=[prompt_status_state, prompt_status, prompt_retry, synthesis_btn],
            ).then(
                fn=_transcribe_prompt,
                inputs=[prompt_audio, prompt_lyric_lang, prompt_vocal_sep],
                outputs=[prompt_metadata, prompt_status_state, prompt_status, prompt_retry],
            )
            prompt_upload.then(
                fn=_update_generate_button,
                inputs=[prompt_metadata, target_metadata], outputs=[synthesis_btn],
            )
            prompt_retry.click(
                fn=_mark_processing,
                outputs=[prompt_status_state, prompt_status, prompt_retry, synthesis_btn],
            ).then(
                fn=_transcribe_prompt,
                inputs=[prompt_audio, prompt_lyric_lang, prompt_vocal_sep],
                outputs=[prompt_metadata, prompt_status_state, prompt_status, prompt_retry],
            ).then(
                fn=_update_generate_button,
                inputs=[prompt_metadata, target_metadata], outputs=[synthesis_btn],
            )

            target_audio.change(
                fn=_clear_target_state,
                inputs=[target_audio, skip_clear_metadata_count],
                outputs=[target_metadata, skip_clear_metadata_count, target_status_state, target_status, target_retry, synthesis_btn],
            )
            target_upload = target_audio.upload(
                fn=_mark_processing,
                outputs=[target_status_state, target_status, target_retry, synthesis_btn],
            ).then(
                fn=_transcribe_target,
                inputs=[target_audio, target_lyric_lang, target_vocal_sep],
                outputs=[target_metadata, target_vocal, target_status_state, target_status, target_retry],
            )
            target_upload.then(
                fn=_update_generate_button,
                inputs=[prompt_metadata, target_metadata], outputs=[synthesis_btn],
            )
            target_retry.click(
                fn=_mark_processing,
                outputs=[target_status_state, target_status, target_retry, synthesis_btn],
            ).then(
                fn=_transcribe_target,
                inputs=[target_audio, target_lyric_lang, target_vocal_sep],
                outputs=[target_metadata, target_vocal, target_status_state, target_status, target_retry],
            ).then(
                fn=_update_generate_button,
                inputs=[prompt_metadata, target_metadata], outputs=[synthesis_btn],
            )

            synthesis_btn.click(
                fn=_mark_result_processing,
                outputs=[result_status_state, result_status, synthesis_btn],
            ).then(
                fn=_run_synthesis,
                inputs=[prompt_audio, prompt_metadata, target_metadata, control_radio, auto_shift, pitch_shift, seed_input],
                outputs=[output_audio, result_status_state, result_status, synthesis_btn],
            )

    return page


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860, help="Gradio server port")
    parser.add_argument("--server-name", default="127.0.0.1", help="Address for the local Gradio server")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 for SVS model and inference")
    args = parser.parse_args()

    page = render_interface()
    page.queue()
    page.launch(share=args.share, server_name=args.server_name, server_port=args.port)
