#![feature(rustc_private)]

extern crate clap;
extern crate memmap;
extern crate rayon;

use std::time::Instant;
use std::{fs, mem, slice};

use std::{f32, u32};
use core::f32::consts::PI;

use memmap::MmapOptions;

use clap::{App, Arg};

use rustfft::FFTplanner;
use rustfft::num_complex::Complex32;
use rayon::prelude::*;

use gnuplot::*;

fn waterfall_image(name: &str, data: &mut [f32], nfft: u32, nseg: u32, cf : f32, bw : f32, twin : f32, plot_psd : bool, background_spectrum : &[f32], min : f32, max : f32) {
    let mut freq = vec![0.0; nfft as usize];
    let mut view_ratio : f64 = 1.0;

    for i in 0..nfft {
        freq[i as usize] = ((bw / nfft as f32) * i as f32) - (bw / 2.0);
    }

    for i in 0..data.len() {
        if data[i] < min {
            data[i] = min;
        }
        if data[i] > max {
            data[i] = max;
        }
    }

    let mut fg = Figure::new();
    if plot_psd {
        // Make averaged background noise profile

        view_ratio = 0.8;
        fg.axes2d()
            .set_pos(0.0, view_ratio)
            .set_size(1.0, 1.0 - view_ratio)
	        .set_title(&format!("Background noise plot"), &[])
	        .set_legend(Graph(0.5), Graph(0.9), &[], &[])
	        .set_x_label(&format!("Frequency(Hz) Center: {}", cf) , &[])
	        .set_y_label("PSD(db/Hz)", &[])
            .set_x_range(Fix(freq[0] as f64), Fix(freq[freq.len()-1] as f64))
            //.set_y_range(Fix(-120.0), Fix(0.0))
	        .lines(
                &freq,
	    	    background_spectrum,
	    	    &[Caption("PSD")],
	        );
    }
    fg.axes2d()
        .set_size(1.0, view_ratio)
	    .set_title(name, &[])
	    .set_x_label("Frequency(Hz)", &[])
	    .set_y_label("TIme(s)", &[])
        .set_x_range(Fix(freq[0] as f64), Fix(freq[freq.len()-1] as f64))
        .set_y_range(Fix(0.0), Fix(nseg as f64 * twin as f64))
	    .image(
            &*data,
		    nseg as usize,
            nfft as usize,
            Some((freq[0] as f64, 0.0, freq[freq.len()-1] as f64, nseg as f64 * twin as f64)),
		    &[Caption("PSD")],
	    );
    println!("Output file: {}", &format!("{}.png", name));
    fg.save_to_png(&format!("{}.png", name), 1024, 4096).expect("Failed writing file");
}

fn main() {
    // Save start time
    let now = Instant::now();

    // Arguments
    let matches = App::new("Rust spec")
        .version("0.0.1")
        .author("mike7c2 <mike7c2@gmail.com>")
        .about("Just a toy")
        .arg(Arg::with_name("input").help("Input data").index(1))
        .arg(
            Arg::with_name("nfft")
                .short("n")
                .long("nfft")
                .value_name("int")
                .takes_value(true)
                .help("FFT Size"),
        )
        .arg(
            Arg::with_name("i")
                .short("i")
                .takes_value(false)
                .help("Input is i8"),
        )
        .arg(
            Arg::with_name("average")
                .short("a")
                .long("average")
                .value_name("int")
                .takes_value(true)
                .help("Number of frame to average for each image line"),
        )
        .arg(
            Arg::with_name("filter")
                .short("f")
                .long("filter")
                .takes_value(false)
                .help("Enable background filtering"),
        )
        .arg(
            Arg::with_name("bw")
                .short("b")
                .long("bw")
                .takes_value(true)
                .help("Bandwidth of input data"),
        )
        .arg(
            Arg::with_name("cf")
                .short("c")
                .long("cf")
                .takes_value(true)
                .help("Center frequency of input data"),
        )
        .arg(
            Arg::with_name("dc")
                .short("d")
                .long("dc")
                .takes_value(true)
                .help("Down conversion factor"),
        )
        .arg(
            Arg::with_name("min")
                .short("n")
                .long("min")
                .takes_value(true)
                .help("Min value for spectrogram"),
        )
        .arg(
            Arg::with_name("max")
                .short("n")
                .long("max")
                .takes_value(true)
                .help("Max value for spectrogram"),
        )
        .get_matches();

    // Variables for arguments and parameters
    let mut nfft = 65536;
    let mut average = 1;
    let mut filter_background = false;
    let mut bw = 240000;
    let mut cf = 100000000;
    let mut nseg = 0;
    let mut input_is_float = true;
    let mut dc = 1;
    let mut min = -100.0;
    let mut max = -40.0;

    // Read arguments
    if let Some(o) = matches.value_of("nfft") {
        nfft = o.parse::<usize>().unwrap();
    }

    if let Some(o) = matches.value_of("average") {
        average = o.parse::<usize>().unwrap();
    }

    if let Some(o) = matches.value_of("bw") {
        bw = o.parse::<u64>().unwrap();
    }

    if let Some(o) = matches.value_of("cf") {
        cf = o.parse::<u64>().unwrap();
    }

    if matches.is_present("filter") {
        filter_background = true;
    }

    if matches.is_present("i") {
        input_is_float = false;
    }

    if let Some(o) = matches.value_of("dc") {
        dc = o.parse::<usize>().unwrap();
    }

    if let Some(o) = matches.value_of("min") {
        min = o.parse::<f32>().unwrap();
    }

    if let Some(o) = matches.value_of("max") {
        max = o.parse::<f32>().unwrap();
    }

    // Time for each window
    let twin = (1.0 / (bw as f32 / nfft as f32)) * average as f32;

    if let Some(o) = matches.value_of("input") {
        // Open file, calculate length and nframes
        let fl = fs::File::open(o).unwrap();
        let mmap: &[u8] = unsafe { &MmapOptions::new().map(&fl).unwrap()[..] };
        let fsamps;
        
        if !input_is_float {
            fsamps = fl.metadata().unwrap().len() / 2;
        } else {
            fsamps = fl.metadata().unwrap().len() / 8;
        }

        // Calculate how many segments to process
        nseg = (fsamps as usize / (nfft * average) - 1) * average;
        println!("Processing file: {}\nFFT Size: {}, Number of windows: {}", o, nfft, nseg);

        // Setup FFT, generate hanning window and setup data buffer
        let fft_plan = FFTplanner::new(false).plan_fft(nfft);
        let fft_window = & ((0..nfft)
                    .into_iter()
                    .map(|i| 0.5 * (1.0 - (2.0 * PI * (i as f32 / ((nfft as f32) - 1.0))).cos()))
                    .collect::<Vec<f32>>());
        let mut psd_data: Vec<f32> = vec![0.0; (nseg * nfft) / average];

        // Calculate FFT scaling
        let window_sum : f32 = fft_window.iter().sum();
        let scale_factor = 1.0 / window_sum;

        // Do all the FFTs, in parallel
        psd_data.par_chunks_mut(nfft)
            .enumerate()
            .map(|(y, x)| {
                let mut inbuf: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); nfft];
                let mut conv : Vec<Complex32> = vec![Complex32::new(0.0, 0.0); nfft];
                let mut outbuf: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); nfft];
                let mut in_data: &[Complex32];

                for i in 0..average {
                    let idx = ((y * average) + i) * nfft;

                    // Process input data chunk, converting if needed
                    if input_is_float {
                        in_data = unsafe {
                            &slice::from_raw_parts_mut(
                                mem::transmute(mmap.as_ptr()),
                                fsamps as usize,
                            )[idx..idx + nfft]
                        };
                    } else {
                        for j in 0..nfft {
                            conv[j] = Complex32::new((mmap[(idx + j) * 2] as f32 - 127.0)/127.0,
                                                     (mmap[(idx + j) * 2 + 1] as f32 - 127.0)/127.0);
                        }
                        in_data = &conv;
                    }

                    // Apply the window
                    inbuf
                       .iter_mut()
                       .zip(in_data)
                       .zip(fft_window)
                       .map(|(a, b)| *a.0 = a.1 * b)
                       .for_each(|_| {});

                    // And do the FFT
                    fft_plan.process(&mut inbuf, &mut outbuf);

                    // Convert to PSD divided by averaging factor, accumulating into output data vector
                    x.iter_mut().zip(&outbuf).map(|(a, b)| *a += 20.0 * (b.norm_sqr() * scale_factor).log10() / average as f32).for_each(|_| {});
                }

                // Mirror each row to make lowest frequency in the center
                let mut tmp: Vec<f32> = vec![0.0; nfft/2];
                for i in 0..nfft/2 {
                    tmp[i] = x[i];
                }
                for i in 0..nfft/2 {
                    x[i] = x[(i+nfft/2)];
                }
                for i in 0..nfft/2 {
                    x[(i+nfft/2)] = tmp[i];
                }
            }).for_each(|_| {});
        println!("FFT Complete");

        // Remove background spectrum noise if requested
        let mut background_spectrum: Vec<f32> = vec![0.0; nfft];
        for i in 0..(nseg/average) {
            for j in 0..nfft {
                background_spectrum[j] += psd_data[i*nfft + j] / (nseg/average) as f32;
            }
        }
        if filter_background {
            for i in 0..(nseg/average) {
                for j in 0..nfft {
                    psd_data[i*nfft + j] -= background_spectrum[j];
                }
            }
        }
        println!("Processing Complete");
        waterfall_image(o, &mut psd_data, nfft as u32, (nseg/average) as u32, cf as f32, bw as f32, twin, true, &background_spectrum, min, max);
    }

    let new_now = Instant::now();
    println!("{:?}", new_now.duration_since(now));
    println!(
        "Processed {} blocks with nfft {}, {}Msamps/s\n",
        nseg,
        nfft,
        (nfft * nseg) as f32 / (new_now.duration_since(now).as_secs_f32() * 1000000.0)
    );
}

