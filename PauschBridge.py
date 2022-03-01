import cv2 as cv
import numpy as np
import os
import random as rd
import yaml

from typing import NewType

bridge_width = 228
bridge_height = 8
frame_rate = 30
codec_code = cv.VideoWriter.fourcc(*'png ')
dtype = 'int16'  # used so we can mask with -1, converted to uint8 that opencv expects before writing

RGB = NewType('RGB', tuple[int, int, int])

Indices = NewType(
    'Indices', tuple[tuple[int, int], tuple[int, int], tuple[int, int]])


def parse_field(data, field, optional=False, default=(0, 0), dtype=int):
    ''' parse yaml field into appropriate tuple values
        :param data:        data dictionary
        :param field:       field to access data dictionary from
        :param optional:    [optional] if True, return default value if field not in data
        :param default:     [optional] value to return if optional flag is true
        :param dtype:       [optional] what to cast tuple vals into (default is integer) '''
    if optional and field not in data:
        return default
    s = data[field]
    s = s.replace('(', '').replace(')', '')
    return tuple(dtype(num) for num in s.split(','))


def parse_sprite_yaml(data, curr_time):
    ''' parses color, position, etc from sprite '''
    params = {}
    params['base_rgb'] = parse_field(data, 'bg_color', True, (-1, -1, -1), int)
    params['highlight_rgb'] = parse_field(data, 'sprite_color')

    for entry in data['positions']:
        params['pos'] = parse_field(entry, 'start')
        params['velocity'] = parse_field(entry, 'velocity', True, dtype=float)
        params['acceleration'] = parse_field(
            entry, 'acceleration', True, dtype=float)
        params['start_time'] = curr_time
        params['end_time'] = params['start_time'] + int(entry['duration'])

        yield params


class PauschFrame:
    def __init__(self):
        self.frame = np.zeros((bridge_height, bridge_width, 3), dtype=dtype)

    def get_base_indices(self):
        return [(0, bridge_height), (0, bridge_width), (0, 3)]

    def get_top(self, indices: Indices = None) -> Indices:
        indices = indices if indices is not None else self.get_base_indices()
        (height_start, height_stop), width, color = indices
        return (height_start, int(height_stop / 2)), width, color

    def get_bottom(self, indices: Indices = None) -> Indices:
        indices = indices if indices is not None else self.get_base_indices()
        (height_start, height_stop), width, color = indices
        return ((height_start - height_stop) / 2, height_stop), width, color

    def get_regions(self, start, indices: Indices = None, end=None) -> Indices:
        # TODO
        frame = mat if mat is not None else self.frame
        return

    def set_values(self, indices: Indices, subframe: np.matrix):
        height, width, rgb = [slice(start, stop) for start, stop in indices]

        mask_data = subframe != -1
        if -1 in subframe:
            print('subframe', sum(sum(sum(subframe != -1))))

        self.frame[height, width, rgb] = np.where(
            mask_data > 0, subframe, self.frame[height, width, rgb])

        #self.frame[height, width, rgb] = subframe


class PauschBridge:
    def __init__(self, num_frames: int = 0):
        self.frames = [PauschFrame() for _ in range(num_frames)]

    def _effect_params(self, start_time: int, end_time: int, slices: list[Indices]):
        ''' boilerplate parameters often needed for any effect methods
            :param start_time:  time (sec) of effect start
            :param end_time:    time (sec) of effect end
            :param slices:      [optional] subset of frame on which the effect takes place
            :return             tuple of start_frame index, end_frame index, and slices '''

        self.add_missing_frames(end_time)
        start_frame = start_time * frame_rate
        end_frame = end_time * frame_rate

        slices = slices if slices is not None else [
            frame.get_base_indices() for frame in self.frames[start_frame:end_frame]]

        return start_frame, end_frame, slices

    def add_missing_frames(self, end_time: int):
        ''' if self.frames is not large enough to incorporate end_time, pad it
            :param end_time: time (sec) to fill self.frames up to'''

        end_index = end_time * frame_rate
        # add missing frames if needed
        if len(self.frames) < end_index:
            self.frames += [PauschFrame()
                            for _ in range(len(self.frames), end_index)]

    def set_values(self, indices: list[Indices], frames: list[np.matrix], start_time, end_time):
        ''' set frame values within the specified timeframe
            :param indices:     subset of frame on which the effect takes place
            :param frames:      frame list to update self.frames, should match size specified by indices
            :param start_time:  time (sec) of effect start
            :param end_time:    time (sec) of effect end '''

        start_frame = start_time * frame_rate
        end_frame = end_time * frame_rate
        print(start_frame, end_frame)
        for inds, mat, frame in zip(indices, frames, range(start_frame, end_frame)):
            self.frames[frame].set_values(inds, mat)

    def get_top(self, start_time, end_time):
        ''' gets list of indices specifying the top half of Pausch Bridge onlu
            :param start_time:  time (sec) of effect start
            :param end_time:    time (sec) of effect end '''

        self.add_missing_frames(end_time)
        # calculate frame indices
        start_index = start_time * frame_rate
        end_index = end_time * frame_rate

        return [frame.get_top() for frame in self.frames[start_index:end_index]]

    def solid_color(self, rgb: RGB, start_time: int, end_time: int, slices: list[Indices] = None):
        ''' effect that displays a solid color on the bridge
            :param rgb:         RGB values of the desired color
            :param start_time:  time (sec) of effect start
            :param end_time:    time (sec) of effect end
            :param slices:      [optional] list of the subset of the frame to display effect on, defaults to whole frame'''

        _, _, slices = self._effect_params(start_time, end_time, slices)

        self.set_values(slices, [rgb for _ in slices], start_time, end_time)

    def hue_shift(self, start_rgb: RGB, end_rgb: RGB, start_time: int, end_time: int, slices: list[Indices] = None):
        ''' effect that displays a gradual (linear) shift from one color to another
            :param start_rgb:   RGB values of the desired starting color
            :param end_rgb:     RGB values of the desired ending color
            :param start_time:  time (sec) of effect start
            :param end_time:    time (sec) of effect end
            :param slices:      [optional] list of the subset of the frame to display effect on, defaults to whole frame'''
        def rgb_ranges(start_rgb: RGB, end_rgb: RGB, num_frames: int):
            ''' generator for hue shift'''
            ranges = [np.linspace(start, end, num_frames)
                      for start, end in zip(start_rgb, end_rgb)]

            for tup in zip(*ranges):
                yield tup

        start_frame, end_frame, slices = self._effect_params(
            start_time, end_time, slices)

        start_frame = start_time * frame_rate
        end_frame = end_time * frame_rate
        num_frames = end_frame - start_frame

        self.set_values(slices, rgb_ranges(
            start_rgb, end_rgb, num_frames), start_time, end_time)

    def sprite_from_file(self, filename: str, start_time: int, end_time: int):
        ''' effect that moves a sprite based on data given from filename
            :param filename:    path to file
            :param start_time:      time (sec) of effect start
            :param end_time:        time (sec) of effect end'''

        # check that file exists
        if not os.path.exists(filename):
            print('filename {} does not exist!'.format(filename))

        # parse actual data
        with open('sprite_data.yaml', 'r') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)

        # each separate entry represents a different sprite
        for sprite_data in data:
            for params in parse_sprite_yaml(sprite_data, start_time):
                print('PARAMS', params)
                self.sprite(**params)
                start_time = params['end_time']

    def sprite(self, highlight_rgb: RGB, start_time: int, end_time: int, pos: tuple[int, int], velocity: tuple[int, int], acceleration: tuple[int, int], base_rgb: RGB, slices: list[Indices] = None):
        ''' effect that displays a small sprite moving linearly
            :param highlight_rgb:   RGB values of the desired sparkle color
            :param start_time:      time (sec) of effect start
            :param end_time:        time (sec) of effect end
            :param pos:             starting position of small sprite
            :param velocity:        velocity of small sprite (2-d tuple)
            :param base_rgb:        [optional] RGB values of the desired base color
            :param slices:          [optional] list of the subset of the frame to display effect on, defaults to whole frame'''

        def gen_slice(pos: tuple[int, int], size: int = 3, limit: tuple[int, int] = (8, 228)):
            x, y = map(round, pos)
            half = size // 2
            min_x = x - half if x - half >= 0 else 0
            min_y = y - half if y - half >= 0 else 0
            max_x = x + half + 1 if x + half + 1 < limit[0] else limit[0]
            max_y = y + half + 1 if y + half + 1 < limit[1] else limit[1]

            return slice(min_x, max_x), slice(min_y, max_y)

        def gen_sprite_movement(num_frames):
            curr_pos = pos
            curr_vel = velocity
            for _ in range(num_frames):
                frame = np.full((bridge_height, bridge_width, 3),
                                base_rgb, dtype=dtype)

                x, y = gen_slice(curr_pos)
                frame[x, y] = highlight_rgb

                curr_vel = [v + a for v, a in zip(curr_vel, acceleration)]
                curr_pos = [p + v for p, v in zip(curr_pos, curr_vel)]

                yield frame

        start_frame, end_frame, slices = self._effect_params(
            start_time, end_time, slices)

        self.set_values(slices, gen_sprite_movement(
            end_frame - start_frame), start_time, end_time)

    def sparkle(self, highlight_rgb: RGB, start_time: int, end_time: int, base_rgb: RGB = (-1, -1, -1), slices: list[Indices] = None):
        ''' effect that displays sparkles of a desired color on a solid background color
            :param highlight_rgb:   RGB values of the desired sparkle color
            :param start_time:      time (sec) of effect start
            :param end_time:        time (sec) of effect end
            :param base_rgb:        [optional] RGB values of the desired base color. If not specified, will not overwrite base color
            :param slices:          [optional] list of the subset of the frame to display effect on, defaults to whole frame'''

        def gen_sparkles(num_frames):
            ''' generator frame function for the sparkles'''
            sparkles = {}
            for frame_i in range(num_frames):
                # gen 15 sparkle every 3 frames
                if not frame_i % 3:
                    for _ in range(15):
                        inds = (rd.randrange(bridge_height),
                                rd.randrange(bridge_width))
                        sparkles[inds] = rd.randrange(3, 7)

                frame = np.full((bridge_height, bridge_width, 3),
                                base_rgb, dtype=dtype)
                for (row, col), value in sparkles.items():
                    if not value:
                        continue

                    sparkles[row, col] -= 1
                    frame[row, col, :] = highlight_rgb

                yield frame

        start_frame, end_frame, slices = self._effect_params(
            start_time, end_time, slices)

        self.set_values(slices, gen_sparkles(
            end_frame - start_frame), start_time, end_time)

    def wave(self, highlight_rgb: RGB, start_time: int, end_time: int, base_rgb: RGB = (-1, -1, -1), slices: list[Indices] = None, width: float = 0.1, speed: int = 30) -> np.matrix:
        ''' effect that displays a wave of desired color & width on a base color
            :param highlight_rgb:   RGB values of the desired wave color
            :param start_time:      time (sec) of effect start
            :param end_time:        time (sec) of effect end
            :param base_rgb:        [optional] RGB values of the desired base color. If not specified, will overlay wave on top of existing color in frames
            :param slices:          [optional] list of the subset of the frame to display effect on, defaults to whole frame
            :param width:           desired width of wave in relation to bridge width, i.e. 0.5 means half the bridge width
            :param speed:           desired speed of wave in pixels / second '''

        def gen_wave(start_frame, end_frame, wave_width):
            dims = tuple([end - start for start, end in slices[0]])
            frame = np.full(dims, base_rgb, dtype=dtype)
            wave_pos = -1
            for _ in range(start_frame, end_frame):
                wave_pos += int(speed / frame_rate)
                wave_start = max(wave_pos - wave_width, 0)
                wave_end = wave_pos
                frame[:, 0:wave_start, :] = base_rgb
                frame[:, wave_start:wave_end, :] = highlight_rgb
                print(
                    'SHAPE', frame[:, wave_start:wave_end, :].shape, highlight_rgb)

                if wave_start >= bridge_width:  # the wave has gone through the whole bridge, start over
                    wave_pos = -1
                yield frame

        start_frame, end_frame, slices = self._effect_params(
            start_time, end_time, slices)

        wave_width = int(width * bridge_width)  # in pixels

        self.set_values(slices, gen_wave(
            start_frame, end_frame, wave_width), start_time, end_time)

    def save(self, basename):
        ''' save frame output to .avi file
            :param basename: base filename (without extension) '''
        filename = basename + '.avi'
        out = cv.VideoWriter(filename, codec_code,
                             frame_rate, (bridge_width, bridge_height))

        for frame in self.frames:
            frame = np.uint8(frame.frame)
            out.write(frame)

        out.release()


def test_wave():
    pbl = PauschBridge()
    pbl.solid_color((255, 0, 0), 0, 10)
    pbl.wave((255, 255, 255), 0, 10)
    pbl.save('test_wave')


def test_sparkle():
    pbl = PauschBridge()
    pbl.solid_color((255, 0, 0), 0, 10)
    pbl.wave((0, 255, 0), 0, 10)
    pbl.sparkle((255, 255, 255), 0, 10)
    pbl.save('test_sparkle')


def test_sprite():
    pbl = PauschBridge()
    pbl.solid_color((255, 0, 0), 0, 10)
    pbl.wave((0, 255, 0), 0, 10)
    pbl.sparkle((255, 255, 255), 0, 10)
    pbl.sprite_from_file('sprite_data.yaml', 0, 5)
    pbl.save('test_sprite')


def simple_test():
    pbl = PauschBridge()
    pbl.solid_color((255, 0, 0), 0, 5)
    pbl.hue_shift((255, 0, 0), (255, 255, 0), 5, 10)
    pbl.solid_color((255, 0, 0), 10, 15)
    pbl.wave((255, 255, 255), 10, 15)
    pbl.sparkle((255, 0, 0), (255, 255, 255), 15, 20)
    pbl.sprite_from_file('sprite_data.yaml', 20, 25)
    pbl.save('test')


def sunset():
    black = (0, 0, 0)
    dark_red = (14, 1, 134)
    yellow = (0, 228, 236)
    sky_blue = (255, 208, 65)
    cloud_grey = (237, 237, 237)
    white = (255, 255, 255)

    pbl = PauschBridge()
    pbl.hue_shift(black, dark_red, 0, 30)
    pbl.hue_shift(dark_red, yellow, 30, 58)
    pbl.hue_shift(yellow, sky_blue, 58, 60)
    pbl.solid_color(sky_blue, 60, 120)
    pbl.wave(sky_blue, cloud_grey, 60, 120, pbl.get_top(60, 120))
    pbl.hue_shift(sky_blue, yellow, 120, 122)
    pbl.hue_shift(yellow, dark_red, 122, 150)
    pbl.hue_shift(dark_red, black, 150, 180)
    pbl.sparkle(black, white, 180, 240)
    pbl.save('sunset')


if __name__ == '__main__':
    # simple_test()
    # sunset()
    # test_wave()
    # test_sparkle()
    test_sprite()
