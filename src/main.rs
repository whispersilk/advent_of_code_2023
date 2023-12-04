use std::fmt::{Display, Formatter, Result as FmtResult};
use std::time::{Duration, SystemTime};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Clone, Copy)]
enum Puzzle {
    First,
    Second,
}

impl Display for Puzzle {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        match self {
            Self::First => write!(f, "1"),
            Self::Second => write!(f, "2"),
        }
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    match args.as_slice() {
        [_] => {
            let start = SystemTime::now();
            for day in 1..=25 {
                let puz1 = run_puzzle(day, Puzzle::First).transpose()?;
                let puz2 = run_puzzle(day, Puzzle::Second).transpose()?;
                if puz1.is_some() || puz2.is_some() {
                    println!("Day {day:2}:");
                }
                if let Some((output, time)) = puz1 {
                    println!("    Part 1 ({time:?}): {output}");
                }
                if let Some((output, time)) = puz2 {
                    println!("    Part 2 ({time:?}): {output}");
                }
            }
            println!("----------------------");
            println!(
                "Total time: {:?}",
                start.elapsed().unwrap_or(Duration::ZERO)
            );
        }
        [_, day, puz] => {
            let day = day.parse::<u32>()?;
            let puzzle = match puz.parse::<u32>()? {
                1 => Puzzle::First,
                2 => Puzzle::Second,
                _ => return Err("[puzzle] can be 1 or 2".into()),
            };
            match run_puzzle(day, puzzle).transpose()? {
                Some((output, time)) => println!("Day {day}, part {puzzle} ({time:?}): {output}"),
                None => println!("Day {day} part {puzzle} is not yet implemented"),
            };
        }
        _ => return Err("Usage: aoc OR aoc [day] [puzzle]".into()),
    };
    Ok(())
}

fn run_puzzle(day: u32, puzzle: Puzzle) -> Option<Result<(String, Duration)>> {
    let start = SystemTime::now();
    let output = Some(match day {
        1 => dec01::run(puzzle),
        2 => dec02::run(puzzle),
        3 => dec03::run(puzzle),
        _ => return None,
    });
    let elapsed = start.elapsed().unwrap_or(Duration::ZERO);
    output.map(|result| result.map(|output| (output, elapsed)))
}

pub mod dec03 {
    use crate::{load_input, Puzzle, Result};

    pub(crate) fn run(puzzle: Puzzle) -> Result<String> {
        match puzzle {
            Puzzle::First => first(),
            Puzzle::Second => second(),
        }
    }

    fn first() -> Result<String> {
        let (parts, symbols) = parse_grid()?;
        let sum = parts
            .iter()
            .filter_map(|part| {
                symbols
                    .iter()
                    .find(|s| part.adjacent_to(s))
                    .map(|_| part.id)
            })
            .sum::<u32>();
        Ok(sum.to_string())
    }

    fn second() -> Result<String> {
        let (parts, symbols) = parse_grid()?;
        let ratio = symbols
            .iter()
            .filter(|symbol| symbol.kind == '*')
            .filter_map(|symbol| {
                let mut iter = parts.iter().filter(|part| part.adjacent_to(symbol));
                match (iter.next(), iter.next(), iter.next()) {
                    (Some(part1), Some(part2), None) => Some(part1.id * part2.id),
                    _ => None,
                }
            })
            .sum::<u32>();
        Ok(ratio.to_string())
    }

    #[derive(Debug)]
    struct Part {
        id: u32,
        row: u32,
        col_left: u32,
        col_right: u32,
    }

    #[derive(Debug)]
    struct Symbol {
        kind: char,
        row: u32,
        col: u32,
    }

    impl Part {
        fn new(id: u32, row: u32, col_left: u32, col_right: u32) -> Self {
            Self {
                id,
                row,
                col_left,
                col_right,
            }
        }

        fn adjacent_to(&self, other: &Symbol) -> bool {
            other.row >= self.row.saturating_sub(1)
                && other.row <= self.row + 1
                && other.col >= self.col_left.saturating_sub(1)
                && other.col <= self.col_right + 1
        }
    }

    fn parse_grid() -> Result<(Vec<Part>, Vec<Symbol>)> {
        let (mut row, mut col) = (0, 0);
        let (mut parts, mut symbols) = (Vec::new(), Vec::new());
        let mut curr_id = None;
        for c in load_input(3)?.chars() {
            match c {
                '\n' => {
                    if let Some(id) = curr_id.take() {
                        parts.push(Part::new(id, row, col - 1 - id.ilog10(), col - 1));
                    };
                    row += 1;
                    col = 0;
                }
                '.' => {
                    if let Some(id) = curr_id.take() {
                        parts.push(Part::new(id, row, col - 1 - id.ilog10(), col - 1));
                    };
                    col += 1;
                }
                '0' | '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9' => {
                    let curr = c as u32 - 48;
                    curr_id = curr_id.map(|v| v * 10 + curr).or(Some(curr));
                    col += 1;
                }
                kind => {
                    if let Some(id) = curr_id.take() {
                        parts.push(Part::new(id, row, col - 1 - id.ilog10(), col - 1));
                    };
                    symbols.push(Symbol { kind, row, col });
                    col += 1;
                }
            };
        }
        parts.sort_unstable_by(|a, b| a.row.cmp(&b.row).then(a.col_left.cmp(&b.col_left)));
        symbols.sort_unstable_by(|a, b| a.row.cmp(&b.row).then(a.col.cmp(&b.col)));
        Ok((parts, symbols))
    }
}

pub mod dec02 {
    use std::convert::TryFrom;

    use crate::{load_input, Puzzle, Result};

    pub(crate) fn run(puzzle: Puzzle) -> Result<String> {
        match puzzle {
            Puzzle::First => first(),
            Puzzle::Second => second(),
        }
    }

    fn first() -> Result<String> {
        let id_sum = load_input(2)?
            .lines()
            .map(Game::try_from)
            .filter(|game| game.as_ref().is_ok_and(|v| v.valid_for(12, 13, 14)))
            .try_fold(0, |acc, x| x.map(|game| acc + game.id))?
            .to_string();
        Ok(id_sum)
    }

    fn second() -> Result<String> {
        let power_sum = load_input(2)?
            .lines()
            .map(Game::try_from)
            .try_fold(0, |acc, x| x.map(|game| acc + game.power()))?
            .to_string();
        Ok(power_sum)
    }

    #[derive(Debug)]
    struct Game {
        pub id: u32,
        pub red: u32,
        pub green: u32,
        pub blue: u32,
    }

    impl Game {
        fn valid_for(&self, r: u32, g: u32, b: u32) -> bool {
            self.red <= r && self.green <= g && self.blue <= b
        }

        fn power(&self) -> u32 {
            self.red * self.green * self.blue
        }
    }

    impl TryFrom<&str> for Game {
        type Error = Box<dyn std::error::Error>;
        fn try_from(value: &str) -> Result<Self> {
            use std::cmp::max;

            let (id, rounds) = value.split_once(':').ok_or(format!("No ':' in {value}"))?;
            let id = id
                .strip_prefix("Game ")
                .ok_or(format!("{id} doesn't start with 'Game '"))?
                .parse::<u32>()?;

            let (red, green, blue) = rounds
                .split(';')
                .map(|round| {
                    let (mut r, mut g, mut b) = (0, 0, 0);
                    for pull in round.split(',') {
                        let (count, color) = pull
                            .trim()
                            .split_once(' ')
                            .ok_or(format!("No ' ' in pull {pull}"))?;
                        let count = count.parse::<u32>()?;
                        match color {
                            "red" => r = count,
                            "green" => g = count,
                            "blue" => b = count,
                            x => Err(format!("Color {x} is invalid"))?,
                        };
                    }
                    Ok((r, g, b))
                })
                .try_fold((0, 0, 0), |a, x: Result<_>| {
                    x.map(|x| (max(a.0, x.0), max(a.1, x.1), max(a.2, x.2)))
                })?;
            Ok(Self {
                id,
                red,
                green,
                blue,
            })
        }
    }
}

pub mod dec01 {
    use crate::{load_input, Puzzle, Result};

    pub(crate) fn run(puzzle: Puzzle) -> Result<String> {
        match puzzle {
            Puzzle::First => first(),
            Puzzle::Second => second(),
        }
    }

    fn first() -> Result<String> {
        let digit_sum = load_input(1)?
            .lines()
            .map(|line| {
                format!(
                    "{}{}",
                    digit(line, Direction::First, NumType::Digit).unwrap(),
                    digit(line, Direction::Last, NumType::Digit).unwrap(),
                )
            })
            .map(|num| num.parse::<u32>().unwrap())
            .sum::<u32>()
            .to_string();
        Ok(digit_sum)
    }

    fn second() -> Result<String> {
        let digit_and_spelled_sum = load_input(1)?
            .lines()
            .map(|line| {
                format!(
                    "{}{}",
                    digit(line, Direction::First, NumType::Either).unwrap(),
                    digit(line, Direction::Last, NumType::Either).unwrap(),
                )
            })
            .map(|num| num.parse::<u32>().unwrap())
            .sum::<u32>()
            .to_string();
        Ok(digit_and_spelled_sum)
    }

    #[derive(Debug)]
    enum Direction {
        First,
        Last,
    }

    #[derive(Debug)]
    enum NumType {
        Digit,
        Either,
    }

    fn digit(line: &str, dir: Direction, kind: NumType) -> Option<u8> {
        let find_digit = |line: &str| {
            let mut iter = line.chars().enumerate().filter(|(_, c)| c.is_ascii_digit());
            let digit = match dir {
                Direction::First => iter.next(),
                Direction::Last => iter.last(),
            };
            digit.map(|(idx, c)| (idx, c.to_digit(10).expect("is digit") as u8))
        };
        let find_spelled = |line: &str| {
            let nums = &[
                "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
            ];
            let outputs = (1..10)
                .zip(nums.iter())
                .filter_map(|(num, spelled)| match dir {
                    Direction::First => line.find(spelled).map(|idx| (idx, num)),
                    Direction::Last => line.rfind(spelled).map(|idx| (idx, num)),
                });
            match dir {
                Direction::First => outputs.min_by_key(|(pos, _)| *pos),
                Direction::Last => outputs.max_by_key(|(pos, _)| *pos),
            }
        };
        match kind {
            NumType::Digit => find_digit(line).map(|(_, n)| n),
            NumType::Either => {
                let options = [find_digit(line), find_spelled(line)];
                let iter = options.iter().filter_map(|n| *n);
                match dir {
                    Direction::First => iter.min_by_key(|(idx, _)| *idx).map(|(_, n)| n),
                    Direction::Last => iter.max_by_key(|(idx, _)| *idx).map(|(_, n)| n),
                }
            }
        }
    }
}

fn load_input(day: u8) -> Result<String> {
    Ok(std::fs::read_to_string(format!("inputs/dec_{day:02}.txt"))?)
}

#[cfg(test)]
mod tests {
    use crate::{run_puzzle, Puzzle, Result};

    #[test]
    fn check_correctness() -> Result<()> {
        let answers = [
            "54159", "53866", // Day 1
            "2317", "74804", // Day 2
            "525119", "76504829", // Day 3
        ];

        for (idx, expected) in answers.iter().map(|s| s.to_string()).enumerate() {
            let day = (idx as u32) / 2 + 1;
            let puzzle = if idx % 2 == 0 {
                Puzzle::First
            } else {
                Puzzle::Second
            };
            match run_puzzle(day, puzzle).transpose()? {
                Some((actual, _)) => assert_eq!(
                    expected, actual,
                    "Failed assertion for day {day}, part {puzzle}."
                ),
                None => panic!(
                    "Solution exists but puzzle is unimplemented for day {day}, part {puzzle}."
                ),
            };
        }
        Ok(())
    }
}
