#!/usr/bin/env python3

import altair as alt
import pandas as pd
import streamlit as st

from colorthief import ColorThief
from matplotlib.pyplot import draw


def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb


@st.cache
def convert_palette(palette):
    df = pd.DataFrame({'index': range(len(palette)), 'colors': palette})
    return df.to_csv(index=False).encode('utf-8')


@st.cache
def get_palette(filename, color_count):
    print('DEBUG: calling get_palette with filename {}, color_count {}'.format(
        filename, color_count))
    color_thief = ColorThief(filename)

    palette = color_thief.get_palette(color_count=color_count, quality=1)
    if len(palette) != color_count:
        palette = color_thief.get_palette(color_count=color_count+1, quality=1)

    s = 'Note: the palette picker had trouble picking {} colors, so it chose {} colors instead'.format(
        color_count, len(palette)) if len(palette) != color_count else ''

    print(palette)
    return palette, s


@st.cache(allow_output_mutation=True)
def draw_palette(palette, s):
    # bgr, not rgb
    # color_thief = ColorThief(filename)

    palette_df = pd.DataFrame({
        'i': range(len(palette)),
        'rgb': [str(tup) for tup in palette],
        'hex': [rgb_to_hex(rgb) for rgb in palette]})

    return alt.Chart(palette_df).mark_rect().encode(
        x=alt.X('i:N', axis=None),
        color=alt.Color('rgb', scale=alt.Scale(
            domain=palette_df.rgb.values, range=palette_df.hex.values), legend=None),
        tooltip=['rgb', 'hex']
    ).properties(
        height=85
    ), s


def st_app():
    st.set_page_config(layout='wide')
    page = st.sidebar.selectbox('Navigation', ['Color picker', 'Presets'])

    if page == 'Color picker':
        st_color_picker_app()
    else:
        st_presets()


def st_presets():
    palettes = {'Rainbow': [(54, 130, 194), (182, 72, 209), (249, 155, 54),
                            (124, 223, 177), (99, 100, 210), (216, 73, 104)],
                'Earthy': [(210, 116, 70), (74, 35, 46), (245, 201, 93), (144, 77, 70), (119, 49, 48), (220, 118, 100)],
                'Spring pastel': [(168, 187, 211), (142, 155, 95), (223, 195, 231), (146, 90, 174), (174, 129, 196), (107, 125, 138)]}

    st.header('Preset palettes')

    left_col, _ = st.columns([1, 2])

    with left_col:
        for name, palette in palettes.items():
            st.subheader(name)
            chart, _ = draw_palette(palette, '')
            st.altair_chart(chart, use_container_width=True)
            st.download_button('Download', convert_palette(
                palette), mime='text/csv')


def st_color_picker_app():
    left_col, right_col = st.columns([2, 1])
    image = st.file_uploader('Upload an image', type=[
        'png', 'jpg', 'jpeg'])
    if not image:
        return

    st.session_state['image'] = image

    if 'color_count' not in st.session_state:
        st.session_state['color_count'] = 6

    with left_col:
        st.subheader('Uploaded image:')
        st.image(st.session_state.image)

    with right_col:
        st.subheader('Generated palatte:')
        st.session_state['color_count'] = st.slider(
            '# colors to generate', 1, 20, 6)
        palette, s = get_palette(
            st.session_state['image'], st.session_state['color_count'])
        chart, s = draw_palette(palette, s)
        st.altair_chart(chart, use_container_width=True)
        if len(s):
            st.write(s)

        st.download_button('Download colors',
                           convert_palette(palette), mime='text/csv')


if __name__ == '__main__':
    st_app()
