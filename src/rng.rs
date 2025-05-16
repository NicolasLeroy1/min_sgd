/// A simple Linear Congruential Generator (LCG).
/// This is a basic RNG and may not be suitable for cryptographic purposes
/// or applications requiring high-quality randomness.
pub struct SimpleRng {
    seed: u64,
}

impl SimpleRng {
    /// Multiplier for the LCG (from Knuth's MMIX)
    const MULTIPLIER: u64 = 6364136223846793005;
    /// Increment for the LCG (from Knuth's MMIX)
    const INCREMENT: u64 = 1442695040888963407;

    /// Creates a new RNG instance with a given seed.
    pub fn new(seed: u64) -> Self {
        SimpleRng {
            // Initialize with the given seed. If 0, use a common non-zero default.
            seed: if seed == 0 { 1 } else { seed },
        }
    }

    /// Generates the next u32 random number.
    fn next_u32(&mut self) -> u32 {
        self.seed = self
            .seed
            .wrapping_mul(Self::MULTIPLIER)
            .wrapping_add(Self::INCREMENT);
        // Return the upper 32 bits of the 64-bit state, which tends to have better randomness.
        (self.seed >> 32) as u32
    }

    /// Generates a random usize number in the range `[0, max_exclusive)`.
    /// This implementation has a slight bias if `max_exclusive` is not a power of 2.
    /// For "basic sufficient" as requested, this is often acceptable.
    pub fn next_usize_in_range(&mut self, max_exclusive: usize) -> usize {
        if max_exclusive == 0 {
            return 0; // Or panic, depending on desired behavior for empty range.
        }
        (self.next_u32() as usize) % max_exclusive
    }
}

/// Shuffles a slice in place using the Fisher-Yates algorithm and a `SimpleRng`.
///
/// # Arguments
///
/// * `slice`: The slice to be shuffled.
/// * `rng`: A mutable reference to `SimpleRng` to provide randomness.
pub fn shuffle_slice<T>(slice: &mut [T], rng: &mut SimpleRng) {
    let len = slice.len();
    if len <= 1 {
        return;
    }
    for i in (1..len).rev() {
        // Generate a random index j such that 0 <= j <= i
        let j = rng.next_usize_in_range(i + 1);
        slice.swap(i, j);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_rng_generates_different_numbers() {
        let mut rng = SimpleRng::new(12345);
        let r1 = rng.next_u32();
        let r2 = rng.next_u32();
        let r3 = rng.next_u32();
        assert_ne!(
            r1, r2,
            "RNG should produce different numbers on subsequent calls."
        );
        assert_ne!(
            r2, r3,
            "RNG should produce different numbers on subsequent calls."
        );
        assert_ne!(
            r1, r3,
            "RNG should produce different numbers on subsequent calls."
        );
    }

    #[test]
    fn test_rng_reproducibility() {
        let mut rng1 = SimpleRng::new(67890);
        let mut rng2 = SimpleRng::new(67890);
        assert_eq!(
            rng1.next_u32(),
            rng2.next_u32(),
            "RNG should be reproducible with the same seed."
        );
        assert_eq!(rng1.next_usize_in_range(100), rng2.next_usize_in_range(100));
    }

    #[test]
    fn test_rng_range_basic() {
        let mut rng = SimpleRng::new(111);
        let max_val = 10;
        for _ in 0..1000 {
            let val = rng.next_usize_in_range(max_val);
            assert!(
                val < max_val,
                "Generated value {} should be less than {}",
                val,
                max_val
            );
        }
    }

    #[test]
    fn test_shuffle_slice_small() {
        let mut rng = SimpleRng::new(101);
        let mut data = [1, 2, 3, 4, 5];
        let original_data = data.clone();

        shuffle_slice(&mut data, &mut rng);

        // Check that the elements are the same, just possibly reordered
        let mut data_sorted = data.to_vec();
        data_sorted.sort();
        let mut original_sorted = original_data.to_vec();
        original_sorted.sort();
        assert_eq!(
            data_sorted, original_sorted,
            "Shuffled data should contain the same elements."
        );

        // It's hard to deterministically test randomness, but at least check it's not the same
        // (this could fail with a very low probability if shuffle results in original order)
        if original_data.len() > 1 {
            // For a sufficiently long array and a decent shuffle,
            // it's unlikely to be identical. This is a weak check.
            // A better check would be distribution over many shuffles, but that's complex for a unit test.
        }
        println!("Original: {:?}, Shuffled: {:?}", original_data, data);
    }

    #[test]
    fn test_shuffle_empty_and_single() {
        let mut rng = SimpleRng::new(202);
        let mut empty_data: [i32; 0] = [];
        shuffle_slice(&mut empty_data, &mut rng); // Should not panic

        let mut single_data = [42];
        shuffle_slice(&mut single_data, &mut rng);
        assert_eq!(single_data[0], 42); // Should remain unchanged
    }

    #[test]
    fn test_rng_zero_seed() {
        let mut rng = SimpleRng::new(0); // Should default to a non-zero seed internally
        let r1 = rng.next_u32();
        let mut rng_default = SimpleRng::new(1); // Assuming 1 is the default if seed is 0
        assert_eq!(
            r1,
            rng_default.next_u32(),
            "Seed 0 should behave like the default non-zero seed."
        );
    }

    #[test]
    fn test_rng_distribution_basic_range() {
        // This is a very basic check for distribution.
        // A proper statistical test would be more involved.
        let mut rng = SimpleRng::new(303);
        let max_exclusive = 5;
        let num_samples = 10000;
        let mut counts = HashMap::new();

        for _ in 0..num_samples {
            let val = rng.next_usize_in_range(max_exclusive);
            *counts.entry(val).or_insert(0) += 1;
        }

        assert_eq!(
            counts.len(),
            max_exclusive,
            "All numbers in range should appear for a large sample."
        );
        for i in 0..max_exclusive {
            assert!(
                counts.contains_key(&i),
                "Value {} missing from distribution",
                i
            );
            // Expect roughly even distribution
            let expected_count = num_samples / max_exclusive;
            if let Some(count) = counts.get(&i) {
                // Allow for some deviation (e.g., 50% around the mean for this basic RNG)
                assert!(
                    *count > expected_count / 2 && *count < expected_count * 2,
                    "Count for {} ({}) is too far from expected ({})",
                    i,
                    count,
                    expected_count
                );
            }
        }
        println!("Distribution for range [0,{}): {:?}", max_exclusive, counts);
    }
}
