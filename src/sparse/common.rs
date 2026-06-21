/// Search `large[cursor..]` for `target` using exponential probes then halving.
/// Updates `cursor` to the insertion point. Returns `Some(pos)` if found, `None` otherwise.
#[must_use]
pub(crate) fn galloping_seek<Idx: Ord + Copy>(
    large: &[Idx],
    cursor: &mut usize,
    target: Idx,
) -> Option<usize> {
    let n = large.len();

    if *cursor >= n {
        return None;
    }

    if *cursor + 1 >= n {
        *cursor = n;

        return None;
    }

    let (last_lt, final_step) = {
        let mut step: usize = 1;
        let mut last_lt = *cursor;
        loop {
            let probe = cursor.saturating_add(step);
            if probe >= n {
                break (last_lt, n - last_lt - 1);
            }
            // SAFETY: `probe < n` is verified by the `if probe >= n` check above,
            // and `large` has length `n`.
            if unsafe { *large.get_unchecked(probe) } >= target {
                break (last_lt, step);
            }
            last_lt = probe;
            step = match step.checked_shl(1) {
                Some(s) => s,
                None => break (last_lt, n - last_lt - 1),
            };
        }
    };

    if final_step == 0 {
        *cursor = n;

        return None;
    }

    let mut step = final_step.next_power_of_two() >> 1;
    let mut pos = last_lt;
    while step > 0 {
        let probe = pos + step;
        // SAFETY: `probe < n` is checked in the same `if`.
        if probe < n && unsafe { *large.get_unchecked(probe) } < target {
            pos = probe;
        }
        step >>= 1;
    }

    let insertion = pos + 1;

    // SAFETY: `insertion < n` is checked in the same `if`.
    if insertion < n && unsafe { *large.get_unchecked(insertion) } == target {
        *cursor = insertion + 1;
        Some(insertion)
    } else {
        *cursor = insertion;
        None
    }
}
