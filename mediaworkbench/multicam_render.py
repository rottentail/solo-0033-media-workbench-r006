"""Stream calibrated camera pictures while preserving the continuous master sound."""
from contextlib import ExitStack
import os
from pathlib import Path
import tempfile
from . import audio_signal, media_sources, multicam_plan, planner, retime_render, sync_clock
from .assets import fingerprint
from .tools import run


def pictures(plan, sources):
    """Yield exactly one materialized picture per programme frame.

    Every frame asks its selected camera's *own* calibration for the source
    position at the absolute programme midpoint, so returning to a camera after
    a cut continues the same action rather than replaying its opening frames.
    A camera whose materialized stream cannot supply the requested frame fails
    loudly; the last frame is never held to fake the missing action.
    """
    size = plan['width'] * plan['height'] * 3
    if len(sources) != len(plan['clips']):
        raise ValueError('camera materializations do not match the plan')
    selected = []
    for shot in plan['shots']:
        for frame in range(shot['start_frame'], shot['end_frame']):
            selected.append((frame, shot['camera']))
    if [frame for frame, _ in selected] != list(range(plan['frames'])):
        raise ValueError('saved shots do not tile the whole programme without gaps')
    for frame, camera_index in selected:
        camera = plan['clips'][camera_index]
        source_sample = sync_clock.position(camera, frame * 1920 + 960)
        if source_sample is None:
            raise ValueError('saved shot has no calibrated source coverage')
        index = int(source_sample // 1920)
        source = sources[camera_index]
        if not 0 <= index < source['frames']:
            raise ValueError('calibrated picture lies outside the recorded camera stream')
        source['handle'].seek(index * size)
        picture = source['handle'].read(size)
        if len(picture) != size:
            raise ValueError('calibrated picture was truncated')
        yield picture


def render(conn, plan_id, destination):
    plan = media_sources.bind(conn, multicam_plan.get(conn, plan_id))
    plan['master'] = media_sources.select(conn, plan['master'])
    planner.verify_sources(dict(clips=plan['clips'] + [plan['master']]))
    destination = Path(destination).expanduser().absolute()
    destination = destination.parent.resolve() / destination.name
    if os.path.lexists(destination) or not destination.parent.is_dir():
        raise ValueError('destination exists or directory is missing')
    with ExitStack() as stack:
        directory = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix='.multicam-', dir=destination.parent)))
        # One bounded disk materialization per camera occurrence; each stream is
        # decoded exactly once, then sampled with bounded random reads.
        sources, unused_audio = retime_render.materialize(plan, directory, stack)
        for shot in plan['shots']:
            camera_index = shot['camera']
            camera = plan['clips'][camera_index]
            source = sources[camera_index]
            for frame in range(shot['start_frame'], shot['end_frame']):
                source_sample = sync_clock.position(camera, frame * 1920 + 960)
                if source_sample is None or int(source_sample // 1920) >= source['frames']:
                    raise ValueError('camera stops recording before its saved shot ends')
        raw = directory / 'programme.rgb'
        with raw.open('wb') as handle:
            for picture in pictures(plan, sources):
                handle.write(picture)
        master = stack.enter_context(audio_signal.materialized(plan['master']))
        output = directory / 'result.mp4'
        run('ffmpeg', ['-v', 'error', '-y', '-threads', '1', '-filter_threads', '1',
            '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '640x360', '-r', '25', '-i', str(raw),
            '-f', 'f32le', '-ar', '48000', '-ac', '2', '-i', str(master),
            '-t', str(plan['samples'] / 48000), '-c:v', 'libx264', '-threads', '1',
            '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-movflags', '+faststart', str(output)])
        planner.verify_sources(dict(clips=plan['clips'] + [plan['master']]))
        checksum = fingerprint(output)
        with output.open('rb') as handle:
            os.fsync(handle.fileno())
        os.link(output, destination)
    return {'plan_id': plan_id, 'output_path': str(destination), 'sha256': checksum,
            'frames': plan['frames'], 'shots': plan['shots'], 'score': plan['score'],
            'cut_count': plan.get('cut_count', len(plan['shots']) - 1)}
