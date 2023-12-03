type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Clone, Copy)]
enum Puzzle {
    First,
    Second,
}

impl std::fmt::Display for Puzzle {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
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
            for day in 1..=25 {
                let puz1 = run_puzzle(day, Puzzle::First).transpose()?;
                let puz2 = run_puzzle(day, Puzzle::Second).transpose()?;
                if puz1.is_some() || puz2.is_some() {
                    println!("Day {day:2}:");
                }
                if let Some(output) = puz1 {
                    println!("    Part 1: {output}");
                }
                if let Some(output) = puz2 {
                    println!("    Part 2: {output}");
                }
            }
        }
        [_, day, puz] => {
            let day = day.parse::<u32>()?;
            let puzzle = match puz.parse::<u32>()? {
                1 => Puzzle::First,
                2 => Puzzle::Second,
                _ => return Err("[puzzle] can be 1 or 2".into()),
            };
            match run_puzzle(day, puzzle).transpose()? {
                Some(output) => println!("Day {day}, part {puzzle}: {output}"),
                None => println!("Day {day} part {puzzle} is not yet implemented"),
            };
        }
        _ => return Err("Usage: aoc OR aoc [day] [puzzle]".into()),
    };
    Ok(())
}

fn run_puzzle(day: u32, puzzle: Puzzle) -> Option<Result<String>> {
    Some(match day {
        1 => dec01::run(puzzle),
        2 => dec02::run(puzzle),
        // 3 => dec03::run(puzzle),
        _ => return None,
    })
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
        unimplemented!()
    }

    fn second() -> Result<String> {
        unimplemented!()
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

    enum Direction {
        First,
        Last,
    }

    #[allow(dead_code)]
    enum NumType {
        Digit,
        Spelled,
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
            NumType::Spelled => find_spelled(line).map(|(_, n)| n),
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
    Ok(std::fs::read_to_string(format!(
        "inputs/dec_{:02}.txt",
        day
    ))?)
}
