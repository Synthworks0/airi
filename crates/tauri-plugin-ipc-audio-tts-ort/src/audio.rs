use anyhow::Result;
use hound::{WavSpec, WavWriter};
use std::io::Cursor;

pub fn to_wav(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>> {
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut cursor = Cursor::new(Vec::new());
    {
        let mut writer = WavWriter::new(&mut cursor, spec)?;

        for &sample in samples {
            // Convert float32 to int16
            let sample_i16 = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
            writer.write_sample(sample_i16)?;
        }

        writer.finalize()?;
    }

    Ok(cursor.into_inner())
}

pub fn apply_pitch_shift(samples: &mut [f32], pitch_factor: f32) {
    if (pitch_factor - 1.0).abs() < 0.01 {
        return; // No significant pitch change
    }

    // Simple pitch shifting using resampling
    // This is a placeholder - real implementation would use a proper pitch shifting algorithm
    let factor = 2.0_f32.powf(pitch_factor / 12.0);

    if factor > 1.0 {
        // Higher pitch - speed up
        let step = factor;
        let mut write_idx = 0;
        let mut read_idx = 0.0;

        while read_idx < samples.len() as f32 - 1.0 {
            let idx = read_idx as usize;
            let frac = read_idx - idx as f32;

            // Linear interpolation
            let sample = samples[idx] * (1.0 - frac) + samples[idx + 1] * frac;
            samples[write_idx] = sample;

            write_idx += 1;
            read_idx += step;

            if write_idx >= samples.len() {
                break;
            }
        }

        // Fill remaining with zeros
        for i in write_idx..samples.len() {
            samples[i] = 0.0;
        }
    }
}

pub fn apply_speed_change(samples: Vec<f32>, speed_factor: f32, sample_rate: u32) -> Result<Vec<f32>> {
    if (speed_factor - 1.0).abs() < 0.01 {
        return Ok(samples); // No significant speed change
    }

        // Use rubato for high-quality resampling
    use rubato::{FftFixedIn, Resampler};

    let new_rate = (sample_rate as f32 * speed_factor) as u32;
    let mut resampler = FftFixedIn::<f32>::new(
        sample_rate as usize,
        new_rate as usize,
        1024,
        1,
        1,
    )?;

    let waves_in = vec![samples];
    let waves_out = resampler.process(&waves_in, None)?;

    Ok(waves_out[0].clone())
}

pub fn apply_volume(samples: &mut [f32], volume_db: f32) {
    if volume_db.abs() < 0.01 {
        return; // No significant volume change
    }

    let gain = 10.0_f32.powf(volume_db / 20.0);

    for sample in samples.iter_mut() {
        *sample *= gain;
        *sample = sample.clamp(-1.0, 1.0); // Prevent clipping
    }
}
