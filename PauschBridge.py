import cv2 as cv
import numpy as np
from typing import NewType
import random as rd

bridge_width = 228
bridge_height = 8
frame_rate = 30
codec_code = cv.VideoWriter.fourcc(*'png ')

RGB = NewType('RGB', tuple[int, int, int])

Indices = NewType(
    'Indices', tuple[tuple[int, int], tuple[int, int], tuple[int, int]])


class PauschFrame:
    def __init__(self):
        self.frame = np.zeros((bridge_height, bridge_width, 3), dtype='uint8')

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
        self.frame[height, width, rgb] = subframe


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

        return [frame.get_top() for index, frame in enumerate(self.frames[start_index:end_index])]

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

    def sparkle(self, base_rgb: RGB, highlight_rgb: RGB, start_time: int, end_time: int, slices: list[Indices] = None):
        ''' effect that displays sparkles of a desired color on a solid background color
            :param base_rgb:        RGB values of the desired base color
            :param highlight_rgb:   RGB values of the desired sparkle color
            :param start_time:      time (sec) of effect start
            :param end_time:        time (sec) of effect end
            :param slices:          [optional] list of the subset of the frame to display effect on, defaults to whole frame'''

        def gen_sparkles(num_frames):
            ''' generator frame function for the sparkles'''
            sparkles = {}
            for frame_i in range(num_frames):
                # gen 5 sparkle every 5 frames
                if not frame_i % 3:
                    for _ in range(15):
                        inds = (rd.randrange(bridge_height),
                                rd.randrange(bridge_width))
                        sparkles[inds] = rd.randrange(3, 7)

                frame = np.full((bridge_height, bridge_width, 3),
                                base_rgb, dtype='uint8')
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

    def wave(self, base_rgb: RGB, highlight_rgb: RGB, start_time: int, end_time: int, slices: list[Indices] = None, width: float = 0.1, speed: int = 30) -> np.matrix:
        ''' effect that displays a wave of desired color & width on a base color
            :param base_rgb:        RGB values of the desired base color
            :param highlight_rgb:   RGB values of the desired wave color
            :param start_time:      time (sec) of effect start
            :param end_time:        time (sec) of effect end
            :param slices:          [optional] list of the subset of the frame to display effect on, defaults to whole frame
            :param width:           desired width of wave in relation to bridge width, i.e. 0.5 means half the bridge width           
            :param speed:           desired speed of wave in pixels / second '''
        def gen_wave(start_frame, end_frame, wave_width):
            dims = tuple([end - start for start, end in slices[0]])
            frame = np.full(dims, base_rgb, dtype='uint8')
            wave_pos = -1
            for _ in range(start_frame, end_frame):
                wave_pos += int(speed / frame_rate)
                wave_start = max(wave_pos - wave_width, 0)
                wave_end = wave_pos
                frame[:, wave_start:wave_end, :] = highlight_rgb
                frame[:, 0:wave_start, :] = base_rgb

                if wave_start >= bridge_width:  # the wave has gone through the whole bridge
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
            out.write(frame.frame)

        out.release()


def simple_test():
    pbl = PauschBridge()
    pbl.solid_color((255, 0, 0), 0, 5)
    pbl.hue_shift((255, 0, 0), (255, 255, 0), 5, 10)
    pbl.wave((255, 0, 0), (255, 255, 255), 10, 15)
    pbl.sparkle((255, 0, 0), (255, 255, 255), 15, 20)
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
    simple_test()
    sunset()
