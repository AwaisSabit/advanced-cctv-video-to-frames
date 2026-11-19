"""Tests for batch processing of multiple videos."""

from frameforge.batch import process_batch, process_batch_file, read_video_list


def test_read_video_list_skips_blanks_and_comments(tmp_path):
    path = tmp_path / "videos.txt"
    path.write_text("a.mp4\n\n# a comment\nb.mp4\n", encoding="utf-8")
    assert read_video_list(path) == ["a.mp4", "b.mp4"]


def test_process_batch_extracts_from_each_video(tmp_path, sample_video):
    from tests.conftest import moving_rectangle_video

    second = tmp_path / "second.avi"
    moving_rectangle_video(second, count=20, fps=10)
    results = process_batch(
        [str(sample_video), str(second)], tmp_path / "out", every_seconds=1.0
    )
    assert [r["frames"] for r in results] == [3, 2]
    assert (tmp_path / "out" / "sample" / "metadata.json").exists()
    assert (tmp_path / "out" / "second" / "metadata.json").exists()


def test_process_batch_file_reads_the_list(tmp_path, sample_video):
    list_path = tmp_path / "videos.txt"
    list_path.write_text(f"{sample_video}\n", encoding="utf-8")
    results = process_batch_file(list_path, tmp_path / "out", every_seconds=1.0)
    assert len(results) == 1
    assert results[0]["frames"] == 3
