#![feature(type_alias_impl_trait)]
use std::time::Duration;

use sdl3::event::Event;
use sdl3::keyboard::Keycode;
use sdl3::libc::sleep;
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
    vec_size: f32,
}

// type Line = impl Fn(f32) -> f32;
type Line = fn(f32) -> f32;

// This is an ODE of degree one of the form dx/dt = f(x) (ẋ = f(x))
type OrdinaryDegreeOneDiffEq = fn(f32) -> f32;

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

        let mut curr = (beg_x, -line(beg_x));
        while curr.0 < end_x {
            let next = (curr.0 + dx, -line(curr.0 + dx));
            let curr_disp = (
                (curr.0 * x_pixels_per_unit) + x,
                (curr.1 * y_pixels_per_unit) + y,
            );
            let next_disp = (
                (next.0 * x_pixels_per_unit) + x,
                (next.1 * y_pixels_per_unit) + y,
            );
            if !next.0.is_finite() || !next.1.is_finite() {
                curr.0 += dx;
                continue;
            }
            if (next.1 - curr.1).abs() > self.axes_size.1 {
                curr = next;
                continue;
            }
            canvas.draw_line(curr_disp, next_disp).unwrap();
            curr = next;
        }
    }

    fn render_line_domain(
        &self,
        line: Line,
        domain: (f32, f32),
        canvas: &mut Canvas<Window>,
        dx: f32,
        color: Color,
    ) {
        canvas.set_draw_color(color);
        let beg_x = domain.0;
        let end_x = domain.1;

        let y_pixels_per_unit = self.window_size.1 as f32 / self.axes_size.1;
        let y_offset = y_pixels_per_unit * self.graph_center.1;
        let y = (self.window_size.1 as f32 / 2.) + y_offset;

        let x_pixels_per_unit = self.window_size.0 as f32 / self.axes_size.0;
        let x_offset = x_pixels_per_unit * self.graph_center.0;
        let x = (self.window_size.0 as f32 / 2.) - x_offset;

        let mut curr = (beg_x, -line(beg_x));
        while curr.0 < end_x {
            let next = (curr.0 + dx, -line(curr.0 + dx));
            let curr_disp = (
                (curr.0 * x_pixels_per_unit) + x,
                (curr.1 * y_pixels_per_unit) + y,
            );
            let next_disp = (
                (next.0 * x_pixels_per_unit) + x,
                (next.1 * y_pixels_per_unit) + y,
            );
            if !next.0.is_finite() || !next.1.is_finite() {
                curr.0 += dx;
                continue;
            }
            canvas.draw_line(curr_disp, next_disp).unwrap();
            curr = next;
        }
    }

    fn render_vector_field_dxdy(&self, dxdt: OrdinaryDegreeOneDiffEq, canvas: &mut Canvas<Window>) {
        let mut y_0 = 0.0;
        let mut x_0 = 0.0;

        let mut vec = Vec::new();
        while y_0 < self.axes_size.1 / 2.0 {
            while x_0 < self.axes_size.0 / 2.0 {
                vec.push((x_0, y_0));
                x_0 += self.tick.0;
            }
            y_0 += self.tick.1;
            x_0 = 0.0;
        }
        x_0 = -self.tick.0;
        y_0 = 0.0;
        while y_0 < self.axes_size.1 / 2.0 {
            while x_0 >= -self.axes_size.0 / 2.0 {
                vec.push((x_0, y_0));
                x_0 -= self.tick.0;
            }
            y_0 += self.tick.1;
            x_0 = -self.tick.0;
        }
        x_0 = 0.0;
        y_0 = -self.tick.1;
        while y_0 >= -self.axes_size.1 / 2.0 {
            while x_0 < self.axes_size.0 / 2.0 {
                vec.push((x_0, y_0));
                x_0 += self.tick.0;
            }
            y_0 -= self.tick.1;
            x_0 = 0.0;
        }
        x_0 = -self.tick.0;
        y_0 = -self.tick.1;
        while y_0 >= -self.axes_size.1 / 2.0 {
            while x_0 >= -self.axes_size.0 / 2.0 {
                vec.push((x_0, y_0));
                x_0 -= self.tick.0;
            }
            y_0 -= self.tick.1;
            x_0 = -self.tick.0;
        }

        self.render_vec_field_dxdy(dxdt, vec, canvas);
    }

    fn render_vector_field_dxdt<T: Fn(f32) -> f32>(
        &self,
        dxdt: T,
        dt: f32,
        method: impl Fn(f32, &T, f32) -> f32,
        canvas: &mut Canvas<Window>,
        color: Color,
    ) {
        let y_pixels_per_unit = self.window_size.1 as f32 / self.axes_size.1;
        let y_offset = y_pixels_per_unit * self.graph_center.1;
        let y_t = (self.window_size.1 as f32 / 2.) + y_offset;

        let x_pixels_per_unit = self.window_size.0 as f32 / self.axes_size.0;
        let x_offset = x_pixels_per_unit * self.graph_center.0;
        let x_t = (self.window_size.0 as f32 / 2.) - x_offset;

        let x_max =  self.axes_size.1 + self.graph_center.1;
        let x_end = -self.axes_size.1 + self.graph_center.1;

        let t_beg = (self.graph_center.0 - self.axes_size.0 / 2.0).max(0.0);
        let t_end = (self.graph_center.0 + self.axes_size.0 / 2.0).max(0.0);
        // `dxdt` calculates the change in x (the vertical axis) based on the
        // horizontal axis (t)
        let mut vec: Vec<f32> = Vec::new();
        let mut x_val = x_end - (x_end.rem_euclid(self.tick.1));
        while x_val <= x_max {
            vec.push(x_val);
            x_val += self.tick.1;
        }

        let mut t = t_beg - (t_beg.rem_euclid(self.tick.0));
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;

        let mut lines = Vec::new();
        while t < t_end {
            for x in vec.iter_mut() {
                let dx = method(*x, &dxdt, dt);
                let abs = dx.abs();
                min = min.min(abs);
                max = max.max(abs);
                let x_next = *x + dx;

                lines.push((
                    ((t * x_pixels_per_unit) + x_t, (-*x * y_pixels_per_unit) + y_t),
                    (((t + dt) * x_pixels_per_unit) + x_t, (-x_next * y_pixels_per_unit) + y_t),
                    abs
                ));
                *x = x_next;
            }
            t += dt;
        }

        let min_color = Color::RED;
        let max_color = color;

        let color = |x: f32| -> Color {
            let grad_val = (x - min) / (max - min);

            Color::RGB(
                (grad_val * min_color.r as f32) as u8 + ((1. - grad_val) * max_color.r as f32) as u8,
                (grad_val * min_color.g as f32) as u8 + ((1. - grad_val) * max_color.g as f32) as u8,
                (grad_val * min_color.b as f32) as u8 + ((1. - grad_val) * max_color.b as f32) as u8,
            )
        };

        for (t, x, dx) in lines {
            canvas.set_draw_color(color(dx));
            canvas.draw_line(t, x).unwrap();
        }
        
    }
    fn render_vector_field_dxdt_4th(
        &self,
        dxdt: impl Fn(f32) -> f32,
        dt: f32,
        canvas: &mut Canvas<Window>,
        color: Color,
    ) {
        let y_pixels_per_unit = self.window_size.1 as f32 / self.axes_size.1;
        let y_offset = y_pixels_per_unit * self.graph_center.1;
        let y_t = (self.window_size.1 as f32 / 2.) + y_offset;

        let x_pixels_per_unit = self.window_size.0 as f32 / self.axes_size.0;
        let x_offset = x_pixels_per_unit * self.graph_center.0;
        let x_t = (self.window_size.0 as f32 / 2.) - x_offset;

        let x_max =  self.axes_size.1 + self.graph_center.1;
        let x_end = -self.axes_size.1 + self.graph_center.1;

        let t_beg = (self.graph_center.0 - self.axes_size.0 / 2.0).max(0.0);
        let t_end = (self.graph_center.0 + self.axes_size.0 / 2.0).max(0.0);
        // `dxdt` calculates the change in x (the vertical axis) based on the
        // horizontal axis (t)
        let mut vec: Vec<f32> = Vec::new();
        let mut x_val = x_end - (x_end.rem_euclid(self.tick.1));
        while x_val <= x_max {
            vec.push(x_val);
            x_val += self.tick.1;
        }

        let mut t = t_beg - (t_beg.rem_euclid(self.tick.0));
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;

        let mut lines = Vec::with_capacity(vec.len());
        while t < t_end {
            for x in vec.iter_mut() {
                let dx = runge_kutta(*x, &dxdt, dt);
                let abs = dx.abs();
                min = min.min(abs);
                max = max.max(abs);
                let x_next = *x + dx;

                lines.push((
                    ((t * x_pixels_per_unit) + x_t, (-*x * y_pixels_per_unit) + y_t),
                    (((t + dt) * x_pixels_per_unit) + x_t, (-x_next * y_pixels_per_unit) + y_t),
                    abs
                ));
                *x = x_next;
            }
            t += dt;
        }

        let min_color = Color::RED;
        let max_color = color;

        let color = |x: f32| -> Color {
            let grad_val = (x - min) / (max - min);

            Color::RGB(
                (grad_val * min_color.r as f32) as u8 + ((1. - grad_val) * max_color.r as f32) as u8,
                (grad_val * min_color.g as f32) as u8 + ((1. - grad_val) * max_color.g as f32) as u8,
                (grad_val * min_color.b as f32) as u8 + ((1. - grad_val) * max_color.b as f32) as u8,
            )
        };

        for (t, x, dx) in lines {
            canvas.set_draw_color(color(dx));
            canvas.draw_line(t, x).unwrap();
        }
        
    }

    fn render_vec_field_dxdy(
        &self,
        dxdt: OrdinaryDegreeOneDiffEq,
        vec: Vec<(f32, f32)>,
        canvas: &mut Canvas<Window>,
    ) {
        let vec: Vec<_> = vec.iter().map(|&coord| (coord, dxdt(coord.0))).collect();

        let y_pixels_per_unit = self.window_size.1 as f32 / self.axes_size.1;
        let y_offset = y_pixels_per_unit * self.graph_center.1;
        let y = (self.window_size.1 as f32 / 2.) + y_offset;

        let x_pixels_per_unit = self.window_size.0 as f32 / self.axes_size.0;
        let x_offset = x_pixels_per_unit * self.graph_center.0;
        let x = (self.window_size.0 as f32 / 2.) - x_offset;

        let min_color = Color::GREEN;
        let max_color = Color::RED;
        let dx_min = vec.iter().map(|val| val.1.abs()).reduce(f32::min).unwrap();
        let dx_max = vec.iter().map(|val| val.1.abs()).reduce(f32::max).unwrap();

        let color = |x: f32| -> Color {
            let grad_val = (x - dx_min) / (dx_max - dx_min);

            Color::RGB(
                (grad_val * min_color.r as f32) as u8 + ((1. - grad_val) * max_color.r as f32) as u8,
                (grad_val * min_color.g as f32) as u8 + ((1. - grad_val) * max_color.g as f32) as u8,
                (grad_val * min_color.b as f32) as u8 + ((1. - grad_val) * max_color.b as f32) as u8,
            )
        };

        vec.into_iter().for_each(|((x_v, y_v), x_dot)| {
            let theta = f32::atan(x_dot);

            let x_len = self.vec_size * f32::cos(theta);
            let y_len = self.vec_size * f32::sin(theta);

            let (x_0, y_0) = (x_v - x_len / 2.0, y_v - y_len / 2.0);
            let (x_1, y_1) = (x_v + x_len / 2.0, y_v + y_len / 2.0);

            let p0 = ((x_0 * x_pixels_per_unit) + x, (-y_0 * y_pixels_per_unit) + y);
            let p1 = ((x_1 * x_pixels_per_unit) + x, (-y_1 * y_pixels_per_unit) + y);

            canvas.set_draw_color(color(x_dot.abs()));
            canvas.draw_line(p0, p1).unwrap();
        });
    }
}

fn euler_method<T: Fn(f32) -> f32>(x: f32, f: &T, dt: f32) -> f32 {
    f(x) * dt
}

fn improved_euler_method<T: Fn(f32) -> f32>(x: f32, f: &T, dt: f32) -> f32 {
    0.5 * (f(x) + f(x + f(x) * dt)) * dt
}

fn runge_kutta<T: Fn(f32) -> f32>(x: f32, f: &T, dt: f32) -> f32 {
    let k1 = f(x);
    let k2 = f(x + (dt / 2.0) * k1);
    let k3 = f(x + (dt / 2.0) * k2);
    let k4 = f(x + k3 * dt);
    (1.0 / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4) * dt
}

pub fn main() {
    let mut graph = GraphWindow {
        window_size: (1280, 720),
        axes_size: (12., 6.),
        graph_center: (0.0, 0.0),
        tick: (0.25, 0.05),
        tick_size: 6.0,
        vec_size: 0.2,
    };

    let (mut canvas, sdl_context) = graph.to_sdl();

    graph.render_to_canvas(&mut canvas);
    canvas.present();

    let mut event_pump = sdl_context.event_pump().unwrap();
    let mut x_state = 0.0;
    let mut y_state = 0.0;
    let mut ran = false;
    
    // graph.render_to_canvas(&mut canvas);
    // graph.render_vector_field_dxdt(
    //     |x| x.powi(2) - 2.0,
    //     0.01,
    //     runge_kutta,
    //     &mut canvas,
    //     Color::GREEN,
    // );
    // canvas.present();

    // let event_subsystem = sdl_context.event().unwrap();
    // event_subsystem.add_event_watch(|event| {
    //     match event {
    //         Event::Quit { .. }
    //         | Event::KeyDown {
    //             keycode: Some(Keycode::Escape | Keycode::Q),
    //             ..
    //         } => std::process::exit(0),
    //         Event::MouseWheel { y, .. } => {
    //             graph.axes_size.0 *= (1.05f32).powf(-y);
    //             graph.axes_size.1 *= (1.05f32).powf(-y);
    //         }
    //         Event::MouseMotion {
    //             mousestate, x, y, ..
    //         } if mousestate.is_mouse_button_pressed(MouseButton::Left) => {
    //             let x_units_per_pixel = graph.axes_size.0 / graph.window_size.0 as f32;
    //             let y_units_per_pixel = graph.axes_size.1 / graph.window_size.1 as f32;

    //             let x_diff = x - x_state;
    //             let y_diff = y_state - y;

    //             graph.graph_center.0 -= x_diff * x_units_per_pixel;
    //             graph.graph_center.1 -= y_diff * y_units_per_pixel;

    //             x_state = x;
    //             y_state = y;
    //         }
    //         Event::MouseMotion { x, y, .. } => {
    //             x_state = x;
    //             y_state = y;
    //         }
    //         _ => {}
    //     }

    //     graph.render_to_canvas(&mut canvas);
    //     graph.render_vector_field_dxdt(
    //         |x| x.powi(2) - 2.0,
    //         0.01,
    //         runge_kutta,
    //         &mut canvas,
    //         Color::GREEN,
    //     );
    //     canvas.present();
    // });


    'running: loop {
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Escape | Keycode::Q),
                    ..
                } => break 'running,
                Event::MouseWheel { y, .. } => {
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
                _ => {
                    if !ran {
                        ran = true;
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(50));
                    continue;
                }
            }
        }

        graph.render_to_canvas(&mut canvas);
        // graph.render_line(f32::cos, &mut canvas, 0.01, Color::RED);
        // graph.render_line(f32::tan, &mut canvas, 0.01, Color::RED);
        // graph.render_line(f32::tanh, &mut canvas, 0.01, Color::BLUE);
        // graph.render_line(f32::sinh, &mut canvas, 0.01, Color::BLUE);
        // graph.render_line(f32::exp, &mut canvas, 0.01, Color::YELLOW);
        // graph.render_line_domain(f32::ln, (0.05, 1000.), &mut canvas, 0.01, Color::YELLOW);
        // graph.render_line_domain(
        //     |x| f32::sqrt(4.0 + -x.powi(2)),
        //     (-2., 2.),
        //     &mut canvas,
        //     0.01,
        //     Color::GREEN,
        // );
        // graph.render_line_domain(
        //     |x| -f32::sqrt(4.0 + -x.powi(2)),
        //     (-2., 2.),
        //     &mut canvas,
        //     0.01,
        //     Color::GREEN,
        // );
        // graph.render_vector_field_dxdy(|x| f32::exp(-x) * f32::sin(x), &mut canvas);
        // graph.render_vector_field_dxdy(|x| 1.0 - x.powf(14.), &mut canvas);
        // graph.render_vector_field(|x| x / f32::sqrt(4.0 - x.powi(2)), &mut canvas);
        // graph.render_vector_field_dxdy(|x| 2.0 * x, &mut canvas);
        // graph.render_line(|x| x.powi(2) - 4.0, &mut canvas, 0.01, Color::BLUE);
        // graph.render_line(|x| 0.333333 * x.powi(3) - 1.2, &mut canvas, 0.01, Color::BLUE);
        // graph.render_vector_field_dxdt(|x| 1.0 - f32::sin(x.exp()), 0.01, &mut canvas);
        // let s: f32 = 0.5;
        // let a: f32 = 2.;
        // graph.render_vector_field_dxdt(|x| s * (1.0 - x) * x.powf(a) - (1.0 - s) * x * (1.0 - x).powf(a), 0.01, &mut canvas);
        // let r = 0.6;
        // let a = 0.8;
        // let b = 2.1;
        // let eq = |x: f32| x + (-x).exp();
        let dt = 0.01;
        let a = 1.1;
        let b = 0.5;
        let e = 1.0;
        let s = 1.0;
        let Q = 0.0;
        // graph.render_vector_field_dxdt(|x| x * (r - a*(x - b).powi(2)), dt, improved_euler_method, &mut canvas, Color::BLUE);
        // graph.render_vector_field_dxdt(|x| x * (r - a*(x - b).powi(2)), dt, runge_kutta, &mut canvas, Color::GREEN);
        // graph.render_vector_field_dxdt(|x| x.powi(2) - 2.0, dt, runge_kutta, &mut canvas, Color::GREEN);
        graph.render_vector_field_dxdt(|x| (1.0 - (a - b * x)*Q) - e * s * x.powi(4), dt, runge_kutta, &mut canvas, Color::GREEN);
        // graph.render_vector_field_dxdt_4th(|x| x * (r - a*(x - b).powi(2)), dt, &mut canvas, Color::GREEN);
        // graph.render_vector_field_dxdt(eq, 0.001, &mut canvas, Color::BLUE);
        // graph.render_vector_field_dxdt_4th(eq, 0.001, &mut canvas, Color::GREEN);
        // graph.render_line(|x| f32::sin(x / 2.0), &mut canvas, 0.01, Color::BLUE);
        // graph.render_line(|x| f32::sin(x.exp()), &mut canvas, 0.01, Color::BLUE);
        canvas.present();
    }
}
