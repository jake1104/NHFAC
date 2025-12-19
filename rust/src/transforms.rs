use rustfft::FftPlanner;
use num_complex::Complex;

pub struct Transform {
    fft_planner: FftPlanner<f32>,
}

impl Transform {
    pub fn new() -> Self {
        Self { fft_planner: FftPlanner::new() }
    }
    
    pub fn fft_forward(&mut self, signal: &[f32]) -> Vec<Complex<f32>> {
        let n = signal.len();
        let fft = self.fft_planner.plan_fft_forward(n);
        let mut input: Vec<_> = signal.iter()
            .map(|&x| Complex { re: x, im: 0.0 })
            .collect();
        fft.process(&mut input);
        input
    }
    
    pub fn hartley_fast(&mut self, signal: &[f32]) -> Vec<f32> {
        let fft = self.fft_forward(signal);
        // DHT = Re(FFT) - Im(FFT)
        fft.iter().map(|c| c.re - c.im).collect()
    }
}
