from PauschBridge import PauschBridge, read_palette


def test_wave():
    pbl = PauschBridge()
    pbl.solid_color((255, 0, 0), 10)
    pbl.wave((255, 255, 255), 10)
    pbl.save('test_wave')


def test_sparkle():
    pbl = PauschBridge()
    pbl.solid_color((186, 102, 50), 10)
    pbl.wave((23, 9, 16), 10)
    pbl.sparkle((255, 255, 255), 10)
    pbl.save('test_sparkle')


def test_sprite():
    pbl = PauschBridge()
    pbl.solid_color((255, 0, 0), 10)
    pbl.wave((0, 255, 0), 10)
    pbl.sparkle((255, 255, 255), 10)
    pbl.sprite_from_file('sprite_data.yaml', 5)
    pbl.save('test_sprite')


def test_wave_top():
    sky_blue = (255, 208, 65)
    cloud_grey = (237, 237, 237)
    pbl = PauschBridge()
    pbl.solid_color(sky_blue, 5).wave(cloud_grey, 5, slices=pbl.get_top(5))
    pbl.save('top_wave')


def simple_test():
    blue = (255, 0, 0)
    white = (255, 255, 255)
    black = (0, 0, 0)
    cyan = (255, 255, 0)
    pbl = PauschBridge().solid_color(blue, 5)
    pbl += PauschBridge().hue_shift(blue, cyan, 5)
    pbl += PauschBridge().solid_color(blue, 5).wave(white, 5)
    pbl += PauschBridge().sparkle(white, 5, base_rgb=black)
    pbl = PauschBridge().sprite_from_file('sprite_data.yaml', 5)

    pbl.save('test')


def colorblock_test():
    pbl = PauschBridge()
    palette = read_palette('ColorPallate_2022-03-02_09-39-54.csv')
    pbl.color_block(palette, 10)
    pbl.save('test_colorblock')


def region_select_test():
    sky_blue = (255, 208, 65)
    black = (0, 0, 0)
    white = (255, 255, 255)
    pbl = PauschBridge()
    pbl.solid_color(sky_blue, 5).hue_shift(
        black, white, 5, slices=pbl.get_region(5, 40, 80))
    pbl.save('test_region')


if __name__ == '__main__':
    test_wave()
    test_sparkle()
    test_sprite()
    test_wave_top()
    simple_test()
    colorblock_test()
    region_select_test()
