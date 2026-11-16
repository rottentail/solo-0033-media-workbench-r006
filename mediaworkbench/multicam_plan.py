"""Choose a complete calibrated camera sequence with a global cut budget."""
import hashlib
import json
import math
from . import db, media_sources, timeline, sync_clock


def _coverage(cameras, frames):
    """Return contiguous [lo, hi) covered frame ranges, or fail on gaps."""
    ranges = []
    for camera in cameras:
        first, last = sync_clock.bounds(camera)
        # Frame f spans programme samples [f*1920, (f+1)*1920); coverage is a
        # single contiguous span, so derive its frame interval directly.
        lo = max(0, (first + 1919) // 1920)
        hi = min(frames, last // 1920)
        if hi <= lo:
            ranges.append((frames, frames))
            continue
        for frame in range(lo, hi):
            if not sync_clock.covers(camera, frame * 1920, (frame + 1) * 1920):
                raise ValueError('camera coverage must be one continuous programme span')
        ranges.append((lo, hi))
    return ranges


def choose(cameras, frames, minimum, cut_limit, cut_cost):
    """Globally optimal complete camera sequence under hard shot constraints.

    The programme is tiled by shots: a shot keeps one camera over a frame
    span, every shot lasts at least `minimum` frames, the number of switches
    never exceeds `cut_limit`, and each switch pays `cut_cost`. Dynamic
    programming runs over shot boundaries (cut count, ending frame, camera).
    Adding the next shot's score is a prefix-sum range query, and the best
    predecessor is tracked with a running maximum while the ending frame
    sweeps forward, so the whole search is O(cut_limit * cameras^2 * frames)
    with memory independent of the number of candidate paths.

    Ties resolve to the lexicographically first per-frame camera sequence via
    a base-C integer key; all compared sequences have the same frame count.
    """
    if frames <= 0 or minimum <= 0 or cut_limit < 0:
        raise ValueError('no complete camera sequence satisfies shot and coverage limits')
    camera_count = len(cameras)
    base = camera_count
    ranges = _coverage(cameras, frames)
    # A camera that covers a programme frame on the calibration clock must also
    # have a real recorded source frame for its midpoint. Refuse the plan at
    # save time when the stream stops before its calibrated position, instead of
    # discovering the missing picture only while rendering.
    for index, (lo, hi) in enumerate(ranges):
        camera = cameras[index]
        source_frames = camera.get('end_ms', 10 ** 9) // 40
        for frame in range(lo, hi):
            source_sample = sync_clock.position(camera, frame * 1920 + 960)
            if source_sample is None or int(source_sample // 1920) >= source_frames:
                raise ValueError('camera stops recording before its calibrated shot ends')
    values = [[sync_clock.score(camera, frame * 1920 + 960) for frame in range(frames)]
              for camera in cameras]
    prefix = [[0] * (frames + 1) for _ in range(camera_count)]
    for camera in range(camera_count):
        running = 0
        for frame in range(frames):
            running += values[camera][frame]
            prefix[camera][frame + 1] = running

    # powers[p] = base**p ; repeat[c][p] = base-C integer for c repeated p times
    powers = [1] * (frames + 1)
    for p in range(1, frames + 1):
        powers[p] = powers[p - 1] * base
    repeat = [[0] * (frames + 1) for _ in range(camera_count)]
    if base > 1:
        for camera in range(camera_count):
            for p in range(1, frames + 1):
                repeat[camera][p] = camera * (powers[p] - 1) // (base - 1)
    else:
        repeat[0] = [0] * (frames + 1)

    def append(key, camera, length):
        return key * powers[length] + repeat[camera][length]

    # layer[k][f][c] = (best value, lex key) for k cuts ending at frame f on c.
    # back[k][f][c] = (previous ending frame, previous camera) for rebuilding.
    layers = [[[None] * camera_count for _ in range(frames + 1)]
              for _ in range(cut_limit + 1)]
    back = [[[None] * camera_count for _ in range(frames + 1)]
            for _ in range(cut_limit + 1)]

    # Layer zero: a single first shot [0, f) on a camera that covers frame 0.
    for camera, (lo, hi) in enumerate(ranges):
        if lo > 0:
            continue
        for end in range(minimum, min(hi, frames) + 1):
            layers[0][end][camera] = (prefix[camera][end], repeat[camera][end])
            back[0][end][camera] = (0, -1)

    def prefer(candidate, current):
        if current is None:
            return True
        if candidate[0] != current[0]:
            return candidate[0] > current[0]
        return candidate[1] < current[1]

    for cuts in range(cut_limit):
        layer = layers[cuts]
        next_layer = layers[cuts + 1]
        next_back = back[cuts + 1]
        for target in range(camera_count):
            target_lo, target_hi = ranges[target]
            for previous_camera in range(camera_count):
                if previous_camera == target:
                    continue
                # Sweep the new shot's ending frame; maintain the best eligible
                # predecessor ending at or before end-minimum. The prefix term
                # prefix[target][end] is shared, so only (value - prefix) and,
                # on a tie, the lexicographic key decide the champion.
                champion = None  # (adjusted value, predecessor frame, lex key)
                first_end = max(minimum, target_lo + minimum)
                for end in range(first_end, min(target_hi, frames) + 1):
                    candidate_start = end - minimum
                    if candidate_start >= target_lo:
                        predecessor = layer[candidate_start][previous_camera]
                        if predecessor is not None:
                            prev_value, prev_key = predecessor
                            adjusted = prev_value - prefix[target][candidate_start]
                            entry = (adjusted, candidate_start, prev_key)
                            if champion is None or entry[0] > champion[0]:
                                champion = entry
                            elif entry[0] == champion[0]:
                                # Equal full value: compare complete sequences.
                                new_len = end - candidate_start
                                old_len = end - champion[1]
                                new_key = append(prev_key, target, new_len)
                                old_key = append(champion[2], target, old_len)
                                if new_key < old_key:
                                    champion = entry
                    if champion is None:
                        continue
                    length = end - champion[1]
                    value = champion[0] + prefix[target][end] - cut_cost
                    key = append(champion[2], target, length)
                    candidate = (value, key)
                    if prefer(candidate, next_layer[end][target]):
                        next_layer[end][target] = candidate
                        next_back[end][target] = (champion[1], previous_camera)

    finalists = []
    for cuts in range(cut_limit + 1):
        for camera in range(camera_count):
            state = layers[cuts][frames][camera]
            if state is not None:
                finalists.append(((state[0], state[1]), cuts, camera, state))
    if not finalists:
        raise ValueError('no complete camera sequence satisfies shot and coverage limits')
    (_, _), cuts, camera, (value, key) = min(
        finalists, key=lambda item: (-item[3][0], item[3][1]))

    # Rebuild shots backwards through the stored boundaries.
    path_camera = [0] * frames
    end = frames
    current_camera = camera
    current_cuts = cuts
    while current_cuts >= 0:
        previous_end, previous_camera = back[current_cuts][end][current_camera]
        for frame in range(previous_end, end):
            path_camera[frame] = current_camera
        if current_cuts == 0:
            if previous_end != 0 or previous_camera != -1:
                raise ValueError('camera sequence reconstruction lost its opening shot')
            break
        end, current_camera = previous_end, previous_camera
        current_cuts -= 1

    if any(not (ranges[c][0] <= f < ranges[c][1]) for f, c in enumerate(path_camera)):
        raise ValueError('selected camera sequence lost programme coverage')
    recomputed = sum(values[c][f] for f, c in enumerate(path_camera)) - cuts * cut_cost
    if recomputed != value:
        raise ValueError('camera sequence score does not match its frame scores')

    shots = []
    for frame, chosen in enumerate(path_camera):
        if shots and shots[-1]['camera'] == chosen:
            shots[-1]['end_frame'] = frame + 1
        else:
            shots.append({'camera': chosen, 'start_frame': frame, 'end_frame': frame + 1})
    if any(shot['end_frame'] - shot['start_frame'] < minimum for shot in shots):
        raise ValueError('saved camera shot is shorter than the required minimum')
    if len(shots) - 1 != cuts:
        raise ValueError('camera sequence cut count does not match its budget use')
    return shots, value


def create(conn, project_id, cameras, master_asset_id, duration_ms,
           minimum_shot_ms=400, max_cuts=8, cut_cost=0):
    if type(duration_ms) is not int or not 40 <= duration_ms <= 30000 or duration_ms % 40:
        raise ValueError('multicam duration must be a whole frame, at most thirty seconds')
    if type(minimum_shot_ms) is not int or not 40 <= minimum_shot_ms <= duration_ms:
        raise ValueError('invalid minimum shot duration')
    if type(max_cuts) is not int or not 0 <= max_cuts <= 16:
        raise ValueError('invalid cut budget')
    if type(cut_cost) is not int or not 0 <= cut_cost <= 100000:
        raise ValueError('invalid cut cost')
    if not isinstance(cameras, list) or not 1 <= len(cameras) <= 4:
        raise ValueError('one to four independent camera occurrences are required')
    snapshot = timeline.snapshot(conn, project_id)
    clips = []
    for camera in cameras:
        asset = media_sources.get(conn, camera['asset_id'])
        metadata = json.loads(asset['metadata'])
        duration = float(metadata['format']['duration']) * 1000
        if not any(s['codec_type'] == 'video' for s in metadata['streams']):
            raise ValueError('camera must contain video')
        clip = {k: asset[k] for k in ('path', 'fingerprint', 'metadata')}
        clip.update(asset_id=asset['id'], start_ms=0, end_ms=round(duration), gain=1,
                    has_audio=any(s['codec_type'] == 'audio' for s in metadata['streams']),
                    clock=sync_clock.calibrate(camera['anchors'], duration), quality=[])
        previous = 0
        for interval in camera.get('quality', []):
            a, b = interval['start_ms'], interval['end_ms']
            score = interval['score']
            if (type(a) is not int or type(b) is not int or a < previous
                    or b <= a or b > duration_ms or type(score) is not int
                    or not -1000 <= score <= 1000):
                raise ValueError('quality intervals must be ordered and disjoint')
            clip['quality'].append({'start_sample': a * 48, 'end_sample': b * 48, 'score': score})
            previous = b
        clips.append(clip)
    master = media_sources.get(conn, master_asset_id)
    metadata = json.loads(master['metadata'])
    if (not any(s['codec_type'] == 'audio' for s in metadata['streams'])
            or float(metadata['format']['duration']) * 1000 < duration_ms):
        raise ValueError('master recording must cover the whole programme')
    master = {k: master[k] for k in ('id', 'path', 'fingerprint', 'metadata')}
    master.update(asset_id=master.pop('id'), start_ms=0, end_ms=duration_ms, gain=1, has_audio=True)
    frames = duration_ms // 40
    minimum = math.ceil(minimum_shot_ms / 40)
    shots, value = choose(clips, frames, minimum, max_cuts, cut_cost)
    # One explanation row per output frame: the camera actually rendered and
    # the quality score that frame earns, so the saved record accounts for the
    # exact total score the final pictures achieve.
    frame_camera = [0] * frames
    for shot in shots:
        for frame in range(shot['start_frame'], shot['end_frame']):
            frame_camera[frame] = shot['camera']
    frame_scores = [
        {'frame': frame, 'camera': chosen,
         'score': sync_clock.score(clips[chosen], frame * 1920 + 960)}
        for frame, chosen in enumerate(frame_camera)]
    plan = {'project': snapshot['project'], 'clips': clips, 'master': master, 'shots': shots,
            'frames': frames, 'samples': duration_ms * 48, 'width': 640, 'height': 360,
            'minimum_shot_ms': minimum_shot_ms, 'max_cuts': max_cuts, 'cut_cost': cut_cost,
            'score': value, 'cut_count': len(shots) - 1, 'frame_scores': frame_scores}
    payload = json.dumps(plan, sort_keys=True, separators=(',', ':'))
    with db.write(conn):
        cursor = conn.execute('INSERT INTO multicam_plans(project_id,payload,payload_sha256) VALUES (?,?,?)',
            (project_id, payload, hashlib.sha256(payload.encode()).hexdigest()))
    return dict(plan, id=cursor.lastrowid)


def get(conn, plan_id):
    row = conn.execute('SELECT * FROM multicam_plans WHERE id=?', (plan_id,)).fetchone()
    if row is None or hashlib.sha256(row['payload'].encode()).hexdigest() != row['payload_sha256']:
        raise ValueError('multicam plan is missing or damaged')
    return dict(json.loads(row['payload']), id=plan_id)
