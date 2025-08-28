use core::f32;

use sdl3::event::Event;
use sdl3::keyboard::Keycode;
use sdl3::mouse::MouseButton;
use sdl3::pixels::Color;
use sdl3::render::Canvas;
use sdl3::video::Window;
use sdl3::Sdl;

#[derive(Debug)]
pub struct GraphWindow {
    window_size: (u32, u32),
    axes_size: (f32, f32),
    graph_center: (f32, f32),
    tick: (f32, f32),
    tick_size: f32,
}

type Line = fn(f32) -> f32;

impl GraphWindow {
    fn to_sdl(&self) -> (Canvas<Window>, Sdl) {
        let sdl_context = sdl3::init().unwrap();
        let video_subsystem = sdl_context.video().unwrap();

        let window = video_subsystem
            .window("Vector Visualizer", self.window_size.0, self.window_size.1)
            .position_centered()
            .build()
            .unwrap();

        let canvas = window.into_canvas();
        (canvas, sdl_context)
    }

    fn render_to_canvas(&self, canvas: &mut Canvas<Window>) {
        canvas.set_draw_color(Color::BLACK);
        canvas.clear();

        canvas.set_draw_color(Color::WHITE);

        let y_pixels_per_unit = self.window_size.1 as f32 / self.axes_size.1;
        let y_offset = y_pixels_per_unit * self.graph_center.1;
        let y = (self.window_size.1 as f32 / 2.) + y_offset;

        let x_pixels_per_unit = self.window_size.0 as f32 / self.axes_size.0;
        let x_offset = x_pixels_per_unit * self.graph_center.0;
        let x = (self.window_size.0 as f32 / 2.) - x_offset;

        let x_axis = ((0.0, y), (self.window_size.0 as f32, y));
        let y_axis = ((x, 0.0), (x, self.window_size.1 as f32));

        canvas.draw_line(x_axis.0, x_axis.1).unwrap();
        canvas.draw_line(y_axis.0, y_axis.1).unwrap();

        // Render Ticks
        let x_tick_offset =
            (self.graph_center.0 / self.tick.0).trunc() * self.tick.0 * x_pixels_per_unit;
        // The offset the difference of how many self.ticks there
        // are from (y_axis_size / 2) and (graph_center)
        let y_tick_offset =
            (self.graph_center.1 / self.tick.1).trunc() * self.tick.1 * y_pixels_per_unit;

        let mut y_0 = 0.0;
        let y_dsp = y_pixels_per_unit * y_0;
        canvas
            .draw_line(
                (-self.tick_size + x, y_dsp + y - y_tick_offset),
                (
                    self.tick_size + x,
                    y_pixels_per_unit * y_0 + y - y_tick_offset,
                ),
            )
            .unwrap();
        while y_0 < self.axes_size.1 / 2.0 {
            // Add the y tick length
            y_0 += self.tick.1;
            let y_dsp = y_pixels_per_unit * y_0;
            canvas
                .draw_line(
                    (-self.tick_size + x, y_dsp + y - y_tick_offset),
                    (
                        self.tick_size + x,
                        y_pixels_per_unit * y_0 + y - y_tick_offset,
                    ),
                )
                .unwrap();
        }
        y_0 = 0.0;
        while y_0 > -self.axes_size.1 / 2.0 {
            // Add the y tick length
            y_0 -= self.tick.1;
            let y_dsp = y_pixels_per_unit * y_0;
            canvas
                .draw_line(
                    (-self.tick_size + x, y_dsp + y - y_tick_offset),
                    (
                        self.tick_size + x,
                        y_pixels_per_unit * y_0 + y - y_tick_offset,
                    ),
                )
                .unwrap();
        }

        let mut x_0 = 0.0;
        let x_dsp = x_pixels_per_unit * x_0;
        canvas
            .draw_line(
                (x_dsp + x + x_tick_offset, -self.tick_size + y),
                (x_dsp + x + x_tick_offset, self.tick_size + y),
            )
            .unwrap();
        while x_0 < self.axes_size.0 / 2.0 {
            x_0 += self.tick.0;
            let x_dsp = x_pixels_per_unit * x_0;
            canvas
                .draw_line(
                    (x_dsp + x + x_tick_offset, -self.tick_size + y),
                    (x_dsp + x + x_tick_offset, self.tick_size + y),
                )
                .unwrap();
        }
        x_0 = 0.0;
        while x_0 > -self.axes_size.0 / 2.0 {
            x_0 -= self.tick.0;
            let x_dsp = x_pixels_per_unit * x_0;
            canvas
                .draw_line(
                    (x_dsp + x + x_tick_offset, -self.tick_size + y),
                    (x_dsp + x + x_tick_offset, self.tick_size + y),
                )
                .unwrap();
        }
    }

    fn render_line(&self, line: Line, canvas: &mut Canvas<Window>, dx: f32, color: Color) {
        canvas.set_draw_color(color);
        let beg_x = (-self.axes_size.0 / 2.0) + (self.graph_center.0);
        let end_x = (self.axes_size.0 / 2.0) + (self.graph_center.0);

        let y_pixels_per_unit = self.window_size.1 as f32 / self.axes_size.1;
        let y_offset = y_pixels_per_unit * self.graph_center.1;
        let y = (self.window_size.1 as f32 / 2.) + y_offset;

        let x_pixels_per_unit = self.window_size.0 as f32 / self.axes_size.0;
        let x_offset = x_pixels_per_unit * self.graph_center.0;
        let x = (self.window_size.0 as f32 / 2.) - x_offset;

        let mut curr = (beg_x, line(beg_x));
        while curr.0 < end_x {
            let next = (curr.0 + dx, -line(curr.0 + dx));
            let curr_disp = ((curr.0 * x_pixels_per_unit) + x, (curr.1 * y_pixels_per_unit) + y);
            let next_disp = ((next.0 * x_pixels_per_unit) + x, (next.1 * y_pixels_per_unit) + y);
            canvas.draw_line(curr_disp, next_disp).unwrap();
            curr = next;
        }
    }
}

pub fn main() {
    let mut graph = GraphWindow {
        window_size: (1280, 720),
        axes_size: (18., 2.),
        graph_center: (0.0, 0.0),
        tick: (f32::consts::PI / 2.0, 0.5),
        tick_size: 6.0,
    };

    let (mut canvas, sdl_context) = graph.to_sdl();

    graph.render_to_canvas(&mut canvas);
    canvas.present();

    let mut event_pump = sdl_context.event_pump().unwrap();
    let mut x_state = 0.0;
    let mut y_state = 0.0;
    'running: loop {
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Escape | Keycode::Q),
                    ..
                } => break 'running,
                Event::MouseWheel {
                    y,
                    ..
                } => {
                    graph.axes_size.0 *= (1.05f32).powf(-y);
                    graph.axes_size.1 *= (1.05f32).powf(-y);
                }
                Event::MouseMotion {
                    mousestate, x, y, ..
                } if mousestate.is_mouse_button_pressed(MouseButton::Left) => {
                    let x_units_per_pixel = graph.axes_size.0 / graph.window_size.0 as f32;
                    let y_units_per_pixel = graph.axes_size.1 / graph.window_size.1 as f32;

                    let x_diff = x - x_state;
                    let y_diff = y_state - y;

                    graph.graph_center.0 -= x_diff * x_units_per_pixel;
                    graph.graph_center.1 -= y_diff * y_units_per_pixel;

                    x_state = x;
                    y_state = y;
                }
                Event::MouseMotion { x, y, .. } => {
                    x_state = x;
                    y_state = y;
                }
                _ => {}
            }
        }

        graph.render_to_canvas(&mut canvas);
        graph.render_line(f32::cos, &mut canvas, 0.01, Color::RED);
        graph.render_line(f32::sin, &mut canvas, 0.01, Color::BLUE);
        graph.render_line(|x| -f32::sin(x), &mut canvas, 0.01, Color::GREEN);
        graph.render_line(|x| -f32::cos(x), &mut canvas, 0.01, Color::YELLOW);
        canvas.present();
    }
}
