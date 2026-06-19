import pytest
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

pytest.importorskip("streamlit")

from changes import main_ui
from changes.app_settings import AppSettings
from changes.editor import EditorState
from changes.exporters import digitone_events
from changes.importers.compact_progression import compact_progression_to_song_model
from changes.library import SongEntry
from changes.models.song_model import HarmonyEvent, Measure, SongModel


@pytest.mark.parametrize(
    ("section_id", "expected"),
    [
        ("A1", "A1"),
        ("A2", "A2"),
        ("B1", "B1"),
        ("Coda1", "CODA1"),
        ("coda2", "CODA2"),
        ("A__OCC1", "A1"),
        ("A__OCC2", "A2"),
        ("Intro__OCC1", "Intro"),
        ("Intro__OCC2", "Intro2"),
    ],
)
def test_section_filter_label_matches_chord_display_label(section_id: str, expected: str) -> None:
    assert main_ui._display_section_label(section_id) == expected
    assert main_ui._section_filter_label(section_id) == expected


class _SessionState(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


def _measure(
    number: int,
    symbol: str,
    *,
    meter: tuple[int, int] = (4, 4),
    section_id: str | None = None,
) -> Measure:
    numerator, denominator = meter
    length = Fraction(4 * numerator, denominator)
    return Measure(
        number=number,
        section_id=section_id,
        meter_numerator=numerator,
        meter_denominator=denominator,
        absolute_start_quarters=Fraction(0),
        harmony=(
            HarmonyEvent(
                id=f"m{number}_h1",
                symbol=symbol,
                measure_number=number,
                offset_quarters=Fraction(0),
                duration_quarters=length,
            ),
        ),
    )


def _song(*measures: Measure) -> SongModel:
    return SongModel(
        title="Meter UI",
        working_key=None,
        performance_tempo=Fraction(120),
        measures=tuple(measures),
    )


def test_dry_run_result_includes_pattern_change_basis(monkeypatch) -> None:
    monkeypatch.delattr(digitone_events, "pattern_change_basis_payload")
    song = compact_progression_to_song_model(
        {
            "name": "Dry Run",
            "tempo": 120,
            "time_signature": "4/4",
            "sections": [{"name": "A", "progression": [["Cmaj7", "Dm7", "G7", "Cmaj7"]]}],
        }
    )
    monkeypatch.setattr(main_ui.st, "session_state", _SessionState({}))

    result = main_ui._build_dry_run_result(song, song, AppSettings())

    assert result["pattern_change_policy"] == "auto_song_mode"
    assert result["pattern_change"] == 32
    assert result["pattern_change_basis"] == {
        "generated_tracks": "1..8",
        "length": 4,
        "speed": "1/8",
    }


def test_playback_song_uses_selected_song_before_dirty_editor_regenerates_sections(monkeypatch, tmp_path) -> None:
    selected_path = tmp_path / "A Night In Tunisia.song.json"
    selected_song = SongModel(title="A Night In Tunisia", working_key="D", performance_tempo=120, measures=())
    monkeypatch.setattr(
        main_ui.st,
        "session_state",
        _SessionState(
            {
                "_selected_path": selected_path,
                "_library": [
                    SongEntry(path=selected_path, title=selected_song.title, song=selected_song),
                ],
                "_editor_dirty": False,
            }
        ),
    )
    monkeypatch.setattr(main_ui, "_dirty_song", lambda: SongModel(title="Regenerated", working_key=None, performance_tempo=120, measures=()))

    assert main_ui._playback_song() is selected_song


def test_load_song_queues_editor_widget_sync_after_header_widget_exists(monkeypatch) -> None:
    song = SongModel(
        title="Loaded Song",
        working_key="D",
        performance_tempo=96,
        measures=(_measure(1, "Cmaj7", section_id="A__OCC1"),),
    )
    state = _SessionState(
        {
            "editor_title": "Already Instantiated",
            "_section_filter_selected": {"old"},
            "_sf_old": True,
        }
    )
    monkeypatch.setattr(main_ui.st, "session_state", state)

    main_ui._load_song_into_editor(song)

    assert state["editor_title"] == "Already Instantiated"
    assert state["_pending_editor_widget_values"]["editor_title"] == "Loaded Song"

    main_ui._apply_pending_editor_widget_values()

    assert state["editor_title"] == "Loaded Song"
    assert state["editor_tempo"] == 96
    assert state["working_key_input"] == "D"


def test_run_send_respects_bundle_by_section(monkeypatch) -> None:
    calls: list[str] = []
    song = SongModel(title="Song", working_key="C", performance_tempo=120, measures=())

    monkeypatch.setattr(main_ui, "_run_send_bundle_by_section", lambda *_args: calls.append("bundle"))
    monkeypatch.setattr(main_ui, "_run_send_linear_split", lambda *_args: calls.append("linear"))

    main_ui._run_send(song, object(), "DEBUG", main_ui._SEND_MODE_BUNDLE)
    main_ui._run_send(song, object(), "DEBUG", main_ui._SEND_MODE_LINEAR)
    main_ui._run_send(song, object(), "DEBUG", f"{main_ui._ICON_BUNDLE} Bundle by Section")

    assert calls == ["bundle", "linear", "bundle"]


def test_send_mode_label_keeps_plain_internal_value() -> None:
    assert main_ui._send_mode_label(main_ui._SEND_MODE_LINEAR).endswith(main_ui._SEND_MODE_LINEAR)
    assert main_ui._send_mode_label(main_ui._SEND_MODE_BUNDLE).endswith(main_ui._SEND_MODE_BUNDLE)
    assert main_ui._normalize_send_mode(f"{main_ui._ICON_BUNDLE} Bundle by Section") == main_ui._SEND_MODE_BUNDLE


def test_request_rerun_can_reset_song_table_without_clearing_search(monkeypatch) -> None:
    state = _SessionState({"_sl_search": "bilbao", "_songlist_table_reset_token": 4})
    monkeypatch.setattr(main_ui.st, "session_state", state)
    monkeypatch.setattr(main_ui.st, "rerun", lambda: (_ for _ in ()).throw(RuntimeError("rerun")))

    with pytest.raises(RuntimeError, match="rerun"):
        main_ui._request_rerun(reset_song_table=True, clear_song_search=False)

    assert state["_sl_search"] == "bilbao"
    assert state["_songlist_table_reset_token"] == 5
    assert state["_last_rerun_request"]["reset_song_table"] is True


def test_request_rerun_can_clear_song_search(monkeypatch) -> None:
    state = _SessionState({"_sl_search": "bilbao", "_songlist_table_reset_token": 4})
    monkeypatch.setattr(main_ui.st, "session_state", state)
    monkeypatch.setattr(main_ui.st, "rerun", lambda: (_ for _ in ()).throw(RuntimeError("rerun")))

    with pytest.raises(RuntimeError, match="rerun"):
        main_ui._request_rerun(reset_song_table=True, clear_song_search=True)

    assert state["_sl_search"] == ""
    assert state["_songlist_search_value"] == ""
    assert state["_songlist_table_reset_token"] == 5


def test_request_rerun_stores_visible_settings_reason_only_when_passed(monkeypatch) -> None:
    state = _SessionState({"_songlist_table_reset_token": 0})
    monkeypatch.setattr(main_ui.st, "session_state", state)
    monkeypatch.setattr(main_ui.st, "rerun", lambda: (_ for _ in ()).throw(RuntimeError("rerun")))

    with pytest.raises(RuntimeError, match="rerun"):
        main_ui._request_rerun()
    assert "_last_rerun_reason" not in state

    with pytest.raises(RuntimeError, match="rerun"):
        main_ui._request_rerun(reason="visible_settings_changed")
    assert state["_last_rerun_reason"] == "visible_settings_changed"


def test_request_rerun_carries_success_message_until_rendered(monkeypatch) -> None:
    state = _SessionState({"_songlist_table_reset_token": 0})
    messages: list[str] = []
    monkeypatch.setattr(main_ui.st, "session_state", state)
    monkeypatch.setattr(main_ui.st, "rerun", lambda: (_ for _ in ()).throw(RuntimeError("rerun")))
    monkeypatch.setattr(main_ui.st, "success", lambda msg: messages.append(str(msg)))
    monkeypatch.setattr(main_ui.st, "error", lambda msg: None)

    with pytest.raises(RuntimeError, match="rerun"):
        main_ui._request_rerun(success_message="Saved.")

    assert state["_ui_success_message"] == "Saved."
    main_ui._render_pending_ui_messages()
    assert messages == ["Saved."]
    assert "_ui_success_message" not in state


def test_no_explicit_rerun_marker(monkeypatch) -> None:
    state = _SessionState({})
    monkeypatch.setattr(main_ui.st, "session_state", state)

    main_ui._no_explicit_rerun("preview_result_rendered_in_current_run")

    assert state["_last_no_explicit_rerun_reason"] == "preview_result_rendered_in_current_run"


def test_status_slot_does_not_render_when_empty(monkeypatch) -> None:
    rendered: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        main_ui.st,
        "markdown",
        lambda body, unsafe_allow_html=False: rendered.append((str(body), bool(unsafe_allow_html))),
    )

    main_ui._render_status_slot([("warning", None), ("error", "")])

    assert rendered == []


def test_status_slot_escapes_html_and_groups_messages(monkeypatch) -> None:
    rendered: list[str] = []
    monkeypatch.setattr(
        main_ui.st,
        "markdown",
        lambda body, unsafe_allow_html=False: rendered.append(str(body)),
    )

    main_ui._render_status_slot([
        ("warning", "Check settings"),
        ("error", "<b>danger</b>"),
    ])

    assert len(rendered) == 1
    assert "eub-status-slot" in rendered[0]
    assert "eub-status-line-warning" in rendered[0]
    assert "eub-status-line-error" in rendered[0]
    assert "Check settings" in rendered[0]
    assert "&lt;b&gt;danger&lt;/b&gt;" in rendered[0]
    assert "<b>danger</b>" not in rendered[0]


def test_hardware_write_warning_message_tracks_confirm_setting() -> None:
    settings = AppSettings(confirm_before_hardware_write=True)
    assert main_ui._hardware_write_warning(settings) is None

    settings.confirm_before_hardware_write = False
    assert main_ui._hardware_write_warning(settings) == (
        "Hardware write confirmation is disabled. SysEx will be sent immediately."
    )


def test_header_meter_summary_keeps_first_seen_order_without_duplicates() -> None:
    song = _song(
        _measure(1, "Cmaj7", meter=(4, 4)),
        _measure(2, "Dm7", meter=(4, 4)),
        _measure(3, "Fmaj7", meter=(3, 4)),
        _measure(4, "G7", meter=(3, 4)),
        _measure(5, "Cmaj7", meter=(4, 4)),
    )

    assert main_ui._song_meter_summary(song) == "4/4, 3/4"


def test_header_meter_summary_keeps_6_8_before_4_4() -> None:
    song = _song(
        _measure(1, "Dmaj7", meter=(6, 8)),
        _measure(2, "C#m7", meter=(6, 8)),
        _measure(3, "Em7", meter=(4, 4)),
    )

    assert main_ui._song_meter_summary(song) == "6/8, 4/4"


def test_chord_display_initial_meter_with_section_label(monkeypatch) -> None:
    monkeypatch.setattr(main_ui.st, "session_state", _SessionState({"_editor_section_labels": {"initial": "A1"}}))
    state = EditorState(cells=["Cmaj7", "|"], cursor=2)
    song = _song(_measure(1, "Cmaj7", meter=(4, 4), section_id="A1"))

    html = main_ui._chord_display_html(state, song)

    assert '<mark class="section-lbl">A1</mark><mark class="meter-lbl">4/4</mark>||' in html


def test_chord_display_meter_change_with_section_label_order(monkeypatch) -> None:
    monkeypatch.setattr(
        main_ui.st,
        "session_state",
        _SessionState({"_editor_section_labels": {"initial": "A1", 4: "B1"}}),
    )
    state = EditorState(cells=["Cmaj7", "|", "Dm7", "G7", "||", "Cmaj7", "Am7", "|"], cursor=8)
    song = _song(
        _measure(1, "Cmaj7", meter=(4, 4), section_id="A1"),
        _measure(2, "Dm7", meter=(4, 4), section_id="A1"),
        _measure(3, "Cmaj7", meter=(3, 4), section_id="B1"),
    )

    html = main_ui._chord_display_html(state, song)

    assert '<mark class="section-lbl">A1</mark><mark class="meter-lbl">4/4</mark>||' in html
    assert '<mark class="section-lbl">B1</mark><mark class="meter-lbl">3/4</mark>||' in html


def test_chord_display_meter_only_change_before_single_barline(monkeypatch) -> None:
    monkeypatch.setattr(main_ui.st, "session_state", _SessionState({"_editor_section_labels": {}}))
    state = EditorState(cells=["Fmaj7", "|", "Fmaj7", "|", "G7", "|"], cursor=6)
    song = _song(
        _measure(1, "Fmaj7", meter=(4, 4)),
        _measure(2, "Fmaj7", meter=(3, 4)),
        _measure(3, "G7", meter=(3, 4)),
    )

    html = main_ui._chord_display_html(state, song)

    assert '<mark class="meter-lbl">3/4</mark>|' in html


def test_chord_display_missing_barline_mapping_does_not_crash(monkeypatch) -> None:
    monkeypatch.setattr(main_ui.st, "session_state", _SessionState({"_editor_section_labels": {}}))
    state = EditorState(cells=["Cmaj7", "|"], cursor=2)
    song = _song(
        _measure(1, "Cmaj7", meter=(4, 4)),
        _measure(2, "Dm7", meter=(3, 4)),
        _measure(3, "G7", meter=(4, 4)),
    )

    html = main_ui._chord_display_html(state, song)

    assert '<mark class="meter-lbl">4/4</mark>||' in html
    assert '<mark class="meter-lbl">3/4</mark>|' in html


def test_songlist_meter_column_uses_summary_and_is_read_only(monkeypatch) -> None:
    song = _song(
        _measure(1, "Cmaj7", meter=(4, 4)),
        _measure(2, "Dm7", meter=(3, 4)),
        _measure(3, "G7", meter=(4, 4)),
    )
    monkeypatch.setattr(
        main_ui.st,
        "session_state",
        _SessionState(
            {
                "_library": [SongEntry(path=Path("song.song.json"), title="Song", song=song)],
                "_selected_path": None,
                "_songlist_table_reset_token": 0,
                "_settings": AppSettings(),
            }
        ),
    )
    monkeypatch.setattr(main_ui.st, "text_input", lambda *args, **kwargs: "")
    monkeypatch.setattr(main_ui.st, "segmented_control", lambda *args, **kwargs: main_ui._SONG_DISPLAY_CHORD_CELLS)
    captured: dict[str, object] = {}

    def _data_editor(df, *args, **kwargs):
        captured["df"] = df.copy()
        captured["disabled"] = kwargs.get("disabled")
        return df

    monkeypatch.setattr(main_ui.st, "data_editor", _data_editor)
    monkeypatch.setattr(
        main_ui.st,
        "column_config",
        SimpleNamespace(
            CheckboxColumn=lambda *args, **kwargs: object(),
            TextColumn=lambda *args, **kwargs: object(),
            NumberColumn=lambda *args, **kwargs: object(),
        ),
    )

    main_ui._render_songlist(show_import=False)

    assert captured["disabled"] == ["Meter"]
    assert list(captured["df"]["Meter"]) == ["4/4, 3/4"]


def test_songlist_restores_mirrored_search_before_filtering(monkeypatch) -> None:
    bilbao = _song(_measure(1, "Cmaj7"))
    autumn = _song(_measure(1, "Dm7"))
    state = _SessionState(
        {
            "_library": [
                SongEntry(path=Path("bilbao.song.json"), title="Bilbao Song", song=bilbao),
                SongEntry(path=Path("autumn.song.json"), title="Autumn Leaves", song=autumn),
            ],
            "_selected_path": None,
            "_songlist_table_reset_token": 0,
            "_songlist_search_value": "bilbao",
            "_settings": AppSettings(),
        }
    )
    monkeypatch.setattr(main_ui.st, "session_state", state)
    monkeypatch.setattr(main_ui.st, "text_input", lambda *args, **kwargs: state["_sl_search"])
    monkeypatch.setattr(main_ui.st, "segmented_control", lambda *args, **kwargs: main_ui._SONG_DISPLAY_CHORD_CELLS)
    captured: dict[str, object] = {}

    def _data_editor(df, *args, **kwargs):
        captured["df"] = df.copy()
        return df

    monkeypatch.setattr(main_ui.st, "data_editor", _data_editor)
    monkeypatch.setattr(
        main_ui.st,
        "column_config",
        SimpleNamespace(
            CheckboxColumn=lambda *args, **kwargs: object(),
            TextColumn=lambda *args, **kwargs: object(),
            NumberColumn=lambda *args, **kwargs: object(),
        ),
    )

    main_ui._render_songlist(show_import=False)

    assert state["_sl_search"] == "bilbao"
    assert state["_songlist_search_value"] == "bilbao"
    assert list(captured["df"]["Title"]) == ["Bilbao Song"]


def test_song_display_mode_defaults_to_chord_cells() -> None:
    assert main_ui._normalize_song_display_mode(None) == main_ui._SONG_DISPLAY_CHORD_CELLS
    assert main_ui._normalize_song_display_mode("bad") == main_ui._SONG_DISPLAY_CHORD_CELLS
    assert main_ui._normalize_song_display_mode(main_ui._SONG_DISPLAY_CLOUD_GRAPH) == main_ui._SONG_DISPLAY_CLOUD_GRAPH


def test_session_init_restores_song_display_mode_from_settings(monkeypatch) -> None:
    state = _SessionState({"_settings": AppSettings(song_display_mode=main_ui._SONG_DISPLAY_CLOUD_GRAPH)})

    monkeypatch.setattr(main_ui.st, "session_state", state)
    monkeypatch.setattr(main_ui, "_refresh_library", lambda: state.__setitem__("_library", []))

    main_ui._ss_init()

    assert state["_song_display_mode"] == main_ui._SONG_DISPLAY_CLOUD_GRAPH


def test_songlist_display_mode_persists_to_settings(monkeypatch) -> None:
    song = _song(_measure(1, "Cmaj7"))
    settings = AppSettings(song_display_mode=main_ui._SONG_DISPLAY_CHORD_CELLS)
    state = _SessionState(
        {
            "_library": [SongEntry(path=Path("song.song.json"), title="Song", song=song)],
            "_selected_path": None,
            "_songlist_table_reset_token": 0,
            "_settings": settings,
        }
    )
    saved: list[str] = []

    monkeypatch.setattr(main_ui.st, "session_state", state)
    monkeypatch.setattr(main_ui.st, "text_input", lambda *args, **kwargs: "")
    monkeypatch.setattr(main_ui.st, "segmented_control", lambda *args, **kwargs: main_ui._SONG_DISPLAY_CLOUD_GRAPH)
    monkeypatch.setattr(main_ui, "save_settings", lambda settings_arg: saved.append(settings_arg.song_display_mode))
    monkeypatch.setattr(main_ui.st, "data_editor", lambda df, *args, **kwargs: df)
    monkeypatch.setattr(
        main_ui.st,
        "column_config",
        SimpleNamespace(
            CheckboxColumn=lambda *args, **kwargs: object(),
            TextColumn=lambda *args, **kwargs: object(),
            NumberColumn=lambda *args, **kwargs: object(),
        ),
    )

    main_ui._render_songlist(show_import=False)

    assert state["_settings"].song_display_mode == main_ui._SONG_DISPLAY_CLOUD_GRAPH
    assert saved == [main_ui._SONG_DISPLAY_CLOUD_GRAPH]


def test_cloud_graph_shows_disabled_message_without_compiling(monkeypatch) -> None:
    messages: list[str] = []
    monkeypatch.setattr(main_ui.st, "info", lambda message: messages.append(str(message)))
    monkeypatch.setattr(
        main_ui,
        "build_cloud_voice_leading_dataframe",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not compile")),
    )

    main_ui._render_cloud_voice_leading_graph(
        _song(_measure(1, "Cmaj7")),
        AppSettings(cloud_tracks=[None, None, None, None, None, None]),
    )

    assert messages == [
        "Cloud layer is disabled.\nEnable Cloud in Layer Options to view the voice-leading graph."
    ]


def test_cloud_graph_empty_section_selection_skips_compile(monkeypatch) -> None:
    song = _song_with_sections("A", "B")
    path = Path("song.song.json")
    state = _SessionState(
        {
            "_selected_path": path,
            "_section_filter_song_identity": main_ui._section_filter_song_identity(song, path),
            "_section_filter_signature": main_ui._section_filter_signature(song, path),
            "_section_filter_selected": set(),
        }
    )
    messages: list[str] = []
    monkeypatch.setattr(main_ui.st, "session_state", state)
    monkeypatch.setattr(main_ui.st, "info", lambda message: messages.append(str(message)))
    monkeypatch.setattr(
        main_ui,
        "build_cloud_voice_leading_dataframe",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not compile")),
    )

    main_ui._render_cloud_voice_leading_graph(song, AppSettings())

    assert messages == ["No sections selected. Select at least one section to view Cloud graph."]


def test_cloud_graph_uses_effective_filtered_song(monkeypatch) -> None:
    import pandas as pd

    song = _song_with_sections("A", "B")
    path = Path("song.song.json")
    state = _SessionState(
        {
            "_selected_path": path,
            "_section_filter_song_identity": main_ui._section_filter_song_identity(song, path),
            "_section_filter_signature": main_ui._section_filter_signature(song, path),
            "_section_filter_selected": {"B"},
        }
    )
    captured: dict[str, SongModel] = {}

    def _build(song_arg, _settings):
        captured["song"] = song_arg
        return pd.DataFrame({"Voice 1": [60], "Voice 2": [64]}, index=[0])

    monkeypatch.setattr(main_ui.st, "session_state", state)
    monkeypatch.setattr(main_ui, "build_cloud_voice_leading_dataframe", _build)
    monkeypatch.setattr(main_ui.st, "markdown", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(main_ui.st, "altair_chart", lambda *_args, **_kwargs: None)

    main_ui._render_cloud_voice_leading_graph(song, AppSettings())

    assert tuple(m.section_id for m in captured["song"].measures) == ("B",)


def test_cloud_section_boundary_badges_use_digitone_step_numbers() -> None:
    song = _song_with_sections("A", "B")

    assert main_ui._cloud_section_boundary_axis_rows(song, 0, 1) == [
        {"step": 0, "section": "A", "step_label": "1"},
        {"step": 1, "section": "B", "step_label": "2"},
    ]


def test_cloud_voice_leading_chart_uses_step_labels_and_hides_y_labels(monkeypatch) -> None:
    import pandas as pd

    captured: dict[str, object] = {}
    df = pd.DataFrame(
        {
            "Voice 1": [52, 55, 56],
            "Voice 2": [64, 67, 65],
        },
        index=[0, 8, 16],
    )
    song = _song(_measure(1, "Cmaj7", section_id="A"))

    monkeypatch.setattr(
        main_ui.st,
        "altair_chart",
        lambda chart, **kwargs: captured.update({"chart": chart, "kwargs": kwargs}),
    )

    main_ui._render_cloud_voice_leading_chart(df, song)

    spec = captured["chart"].to_dict()
    main_spec = spec["vconcat"][0]
    boundary_spec = spec["vconcat"][1]

    assert main_spec["encoding"]["x"]["axis"]["values"] == [0, 8, 16]
    assert main_spec["encoding"]["x"]["axis"]["labels"] is False
    assert main_spec["encoding"]["x"]["axis"]["ticks"] is False
    assert main_spec["mark"]["interpolate"] == "catmull-rom"
    assert main_spec["encoding"]["y"]["axis"]["labels"] is False
    assert main_spec["encoding"]["y"]["axis"]["ticks"] is False
    assert main_spec["encoding"]["color"]["legend"] is None
    assert main_spec["encoding"]["color"]["scale"]["range"] == main_ui._CLOUD_VOICE_COLOR_RANGE
    assert main_spec["encoding"]["y"]["scale"]["domain"] == [52, 67]
    assert boundary_spec["layer"][0]["mark"]["shape"] == "square"
    assert boundary_spec["layer"][1]["encoding"]["text"]["field"] == "section"
    assert main_spec["height"] + boundary_spec["height"] == main_ui._CLOUD_GRAPH_HEIGHT
    assert captured["kwargs"]["width"] == "stretch"
    assert captured["kwargs"]["height"] == main_ui._CLOUD_GRAPH_HEIGHT


def test_cloud_graph_does_not_render_caption(monkeypatch) -> None:
    import pandas as pd

    song = _song(_measure(1, "Cmaj7"))
    monkeypatch.setattr(main_ui.st, "session_state", _SessionState({}))
    monkeypatch.setattr(
        main_ui,
        "build_cloud_voice_leading_dataframe",
        lambda *_args, **_kwargs: pd.DataFrame({"Voice 1": [60]}, index=[0]),
    )
    monkeypatch.setattr(main_ui.st, "caption", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("caption should not render")))
    monkeypatch.setattr(main_ui.st, "markdown", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(main_ui.st, "altair_chart", lambda *_args, **_kwargs: None)

    main_ui._render_cloud_voice_leading_graph(song, AppSettings())


# ---------------------------------------------------------------------------
# Section filter: empty / stale / signature tests
# ---------------------------------------------------------------------------

def _song_with_sections(*section_ids: str) -> SongModel:
    measures = tuple(
        _measure(i + 1, "Cmaj7", section_id=sid)
        for i, sid in enumerate(section_ids)
    )
    return SongModel(title="Test", working_key="C", performance_tempo=Fraction(120), measures=measures)


# Test 4: empty selection means disabled, not all-selected

def test_action_disabled_reason_empty_selection_is_disabled(monkeypatch) -> None:
    """selected == empty must return a disabled reason, not treat it as all-selected."""
    state = _SessionState({})
    monkeypatch.setattr(main_ui.st, "session_state", state)
    from changes.app_settings import AppSettings
    settings = AppSettings()

    reason = main_ui._action_disabled_reason(
        has_selected_song=True,
        settings=settings,
        selected_sections=set(),
        song_has_sections=True,
    )

    assert reason is not None
    assert "section" in reason.lower()


def test_filtered_song_for_send_empty_selection_not_full_song(monkeypatch) -> None:
    """_filtered_song_for_send with empty selected must not return the full song."""
    path = Path("song.json")
    song = _song_with_sections("S1", "S2")
    state = _SessionState(
        {
            "_selected_path": path,
            "_section_filter_song_identity": main_ui._section_filter_song_identity(song, path),
            "_section_filter_song_path": str(path),
            "_section_filter_signature": main_ui._section_filter_signature(song, path),
            "_section_filter_selected": set(),
        }
    )
    monkeypatch.setattr(main_ui.st, "session_state", state)

    result = main_ui._filtered_song_for_send(song)

    assert len(result.measures) == 0


def test_render_section_filter_requests_rerun_when_selection_changes(monkeypatch) -> None:
    path = Path("song.json")
    song = _song_with_sections("A1", "B1")
    state = _SessionState(
        {
            "_selected_path": path,
            "_section_filter_song_identity": main_ui._section_filter_song_identity(song, path),
            "_section_filter_song_path": str(path),
            "_section_filter_signature": main_ui._section_filter_signature(song, path),
            "_section_filter_selected": {"A1", "B1"},
        }
    )
    monkeypatch.setattr(main_ui.st, "session_state", state)
    monkeypatch.setattr(main_ui.st, "caption", lambda *args, **kwargs: None)

    checkbox_values = iter([True, False])

    class _Column:
        def checkbox(self, *args, **kwargs):
            return next(checkbox_values)

    monkeypatch.setattr(main_ui.st, "columns", lambda *args, **kwargs: [_Column(), _Column()])

    reasons: list[str | None] = []

    def _rerun(*, reason: str | None = None, **kwargs) -> None:
        reasons.append(reason)
        raise RuntimeError("rerun")

    monkeypatch.setattr(main_ui, "_request_rerun", _rerun)

    with pytest.raises(RuntimeError, match="rerun"):
        main_ui._render_section_filter(song)

    assert state["_section_filter_selected"] == {"A1"}
    assert reasons == ["section_filter_changed"]


def test_render_section_filter_does_not_request_rerun_when_selection_is_same(monkeypatch) -> None:
    path = Path("song.json")
    song = _song_with_sections("A1", "B1")
    state = _SessionState(
        {
            "_selected_path": path,
            "_section_filter_song_identity": main_ui._section_filter_song_identity(song, path),
            "_section_filter_song_path": str(path),
            "_section_filter_signature": main_ui._section_filter_signature(song, path),
            "_section_filter_selected": {"A1"},
        }
    )
    monkeypatch.setattr(main_ui.st, "session_state", state)
    monkeypatch.setattr(main_ui.st, "caption", lambda *args, **kwargs: None)

    checkbox_values = iter([True, False])

    class _Column:
        def checkbox(self, *args, **kwargs):
            return next(checkbox_values)

    monkeypatch.setattr(main_ui.st, "columns", lambda *args, **kwargs: [_Column(), _Column()])
    monkeypatch.setattr(
        main_ui,
        "_request_rerun",
        lambda **kwargs: pytest.fail("_request_rerun should not be called"),
    )

    main_ui._render_section_filter(song)

    assert state["_section_filter_selected"] == {"A1"}


# Test 5: stale non-empty section selection resets

def test_stale_nonempty_selection_resets_to_all_sections(monkeypatch) -> None:
    """selected not-empty but not a subset of current sections must reset to all sections."""
    path = Path("song.json")
    song = _song_with_sections("A__OCC1", "B__OCC1")
    old_signature = (str(path), 2, ("S1", "S2"))
    state = _SessionState(
        {
            "_section_filter_song_identity": main_ui._section_filter_song_identity(song, path),
            "_section_filter_song_path": str(path),
            "_section_filter_signature": old_signature,
            "_section_filter_selected": {"S1"},
        }
    )
    monkeypatch.setattr(main_ui.st, "session_state", state)

    result = main_ui._get_or_init_section_filter(song, path)

    assert result == {"A__OCC1", "B__OCC1"}
    assert state["_section_filter_selected"] == {"A__OCC1", "B__OCC1"}


# Test 6: empty selected is preserved across signature change

def test_empty_selected_preserved_when_signature_changes(monkeypatch) -> None:
    """When selected is empty and signature changes, empty must be preserved as user intent."""
    path = Path("song.json")
    song_v2 = _song_with_sections("A__OCC1", "B__OCC1")
    old_signature = (str(path), 1, ("S1",))  # different signature
    state = _SessionState(
        {
            "_section_filter_song_identity": main_ui._section_filter_song_identity(song_v2, path),
            "_section_filter_song_path": str(path),
            "_section_filter_signature": old_signature,
            "_section_filter_selected": set(),
        }
    )
    monkeypatch.setattr(main_ui.st, "session_state", state)

    result = main_ui._get_or_init_section_filter(song_v2, path)

    assert result == set()
    assert state["_section_filter_selected"] == set()


def test_valid_subset_preserved_when_signature_changes(monkeypatch) -> None:
    """A valid subset must be kept even when signature changes (e.g. transpose)."""
    path = Path("song.json")
    song = _song_with_sections("S1", "S2")
    old_signature = (str(path), 2, ("S1", "S2", "S3"))  # different signature
    state = _SessionState(
        {
            "_section_filter_song_identity": main_ui._section_filter_song_identity(song, path),
            "_section_filter_song_path": str(path),
            "_section_filter_signature": old_signature,
            "_section_filter_selected": {"S1"},  # still valid after signature change
        }
    )
    monkeypatch.setattr(main_ui.st, "session_state", state)

    result = main_ui._get_or_init_section_filter(song, path)

    assert result == {"S1"}


def test_new_path_resets_to_all_sections(monkeypatch) -> None:
    """Identity change (new song selected) must reset to all sections."""
    old_path = Path("old_song.json")
    new_path = Path("new_song.json")
    song = _song_with_sections("A__OCC1", "B__OCC1")
    state = _SessionState(
        {
            "_section_filter_song_identity": main_ui._section_filter_song_identity(song, old_path),
            "_section_filter_song_path": str(old_path),
            "_section_filter_signature": (str(old_path), 2, ("X1", "X2")),
            "_section_filter_selected": {"X1"},
        }
    )
    monkeypatch.setattr(main_ui.st, "session_state", state)

    result = main_ui._get_or_init_section_filter(song, new_path)

    assert result == {"A__OCC1", "B__OCC1"}


# Test 7: Transpose after selected section does not produce 0 measures

def test_transpose_with_selected_section_preserves_measures(monkeypatch) -> None:
    """After transpose (override set), _filtered_song_for_send must not return 0 measures."""
    path = Path("song.json")
    library_song = _song_with_sections("S1", "S2")
    from changes.song_filter import transpose_song_model_preserving_structure
    transposed = transpose_song_model_preserving_structure(library_song, lambda s: s, lambda k: k)

    state = _SessionState(
        {
            "_selected_path": path,
            "_section_filter_song_identity": main_ui._section_filter_song_identity(transposed, path),
            "_section_filter_song_path": str(path),
            "_section_filter_signature": main_ui._section_filter_signature(transposed, path),
            "_section_filter_selected": {"S1"},
        }
    )
    monkeypatch.setattr(main_ui.st, "session_state", state)

    result = main_ui._filtered_song_for_send(transposed)

    assert len(result.measures) > 0


# Test 8: section filter signature changes when song structure changes

def test_section_filter_signature_includes_section_ids(monkeypatch) -> None:
    """Signature must differ when section IDs differ (detects dirty-song section change)."""
    path = Path("song.json")
    song_a = _song_with_sections("S1", "S2")
    song_b = _song_with_sections("A__OCC1", "B__OCC1")

    sig_a = main_ui._section_filter_signature(song_a, path)
    sig_b = main_ui._section_filter_signature(song_b, path)

    assert sig_a != sig_b


# ---------------------------------------------------------------------------
# Task 2: Song identity / checkbox namespace / search signature tests
# ---------------------------------------------------------------------------

def test_different_song_identity_resets_filter_even_when_section_ids_match(monkeypatch) -> None:
    """Switching to a different song must reset to all sections even if section IDs are identical."""
    path_a = Path("song_a.json")
    path_b = Path("song_b.json")
    song_a = _song_with_sections("A1", "B1")
    song_b = _song_with_sections("A1", "B1")  # same section IDs, different song

    state = _SessionState(
        {
            "_section_filter_song_identity": main_ui._section_filter_song_identity(song_a, path_a),
            "_section_filter_song_path": str(path_a),
            "_section_filter_signature": main_ui._section_filter_signature(song_a, path_a),
            "_section_filter_selected": {"A1"},  # user unchecked B1 in song A
        }
    )
    monkeypatch.setattr(main_ui.st, "session_state", state)

    result = main_ui._get_or_init_section_filter(song_b, path_b)

    assert result == {"A1", "B1"}


def test_transpose_does_not_change_song_identity() -> None:
    """Transpose preserves path, title, and measure count — identity must be stable."""
    from changes.song_filter import transpose_song_model_preserving_structure

    path = Path("song.json")
    song = _song_with_sections("S1", "S2")
    transposed = transpose_song_model_preserving_structure(song, lambda s: f"T({s})", lambda k: k)

    assert (
        main_ui._section_filter_song_identity(song, path)
        == main_ui._section_filter_song_identity(transposed, path)
    )


def test_section_checkbox_namespace_differs_for_different_paths() -> None:
    """Different song paths must yield different checkbox namespaces."""
    assert (
        main_ui._section_checkbox_namespace(Path("song_a.json"))
        != main_ui._section_checkbox_namespace(Path("song_b.json"))
    )


def test_section_checkbox_namespace_stable_for_same_path() -> None:
    """Same path must always produce the same checkbox namespace."""
    path = Path("my_song.json")
    assert main_ui._section_checkbox_namespace(path) == main_ui._section_checkbox_namespace(path)


def test_clear_section_filter_state_removes_stored_state_and_checkbox_keys(monkeypatch) -> None:
    """_clear_section_filter_state must delete all _sf_* widget keys and stored filter state."""
    state = _SessionState(
        {
            "_sf_abc12345_A1": False,
            "_sf_abc12345_B1": True,
            "_section_filter_selected": {"B1"},
            "_section_filter_song_identity": ("path", "Song", 2),
            "_section_filter_signature": ("path", 2, ("A1", "B1")),
            "_section_filter_song_path": "path",
            "_other_key": "keep_me",
        }
    )
    monkeypatch.setattr(main_ui.st, "session_state", state)

    main_ui._clear_section_filter_state()

    assert "_sf_abc12345_A1" not in state
    assert "_sf_abc12345_B1" not in state
    assert "_section_filter_selected" not in state
    assert "_section_filter_song_identity" not in state
    assert "_section_filter_signature" not in state
    assert state["_other_key"] == "keep_me"


def test_song_table_search_signature_differs_for_different_text() -> None:
    """Different search strings must produce different signatures."""
    assert (
        main_ui._song_table_search_signature("autumn leaves")
        != main_ui._song_table_search_signature("stella by starlight")
    )


def test_song_table_search_signature_normalises_case_and_whitespace() -> None:
    """Signature must be case-insensitive and strip leading/trailing whitespace."""
    assert main_ui._song_table_search_signature("Autumn") == main_ui._song_table_search_signature("autumn")
    assert main_ui._song_table_search_signature("  autumn  ") == main_ui._song_table_search_signature("autumn")


def test_song_table_search_signature_empty_and_blank_are_equal() -> None:
    """Empty string and whitespace-only search must yield the same signature."""
    assert main_ui._song_table_search_signature("") == main_ui._song_table_search_signature("   ")
    assert main_ui._song_table_search_signature(None) == main_ui._song_table_search_signature("")
