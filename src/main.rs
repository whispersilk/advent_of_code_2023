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
        [_, day] => {
            let day = day.parse::<u32>()?;
            for puzzle in [Puzzle::First, Puzzle::Second] {
                match run_puzzle(day, puzzle).transpose()? {
                    Some((output, time)) => {
                        println!("Day {day}, part {puzzle} ({time:?}): {output}")
                    }
                    None => println!("Day {day} part {puzzle} is not yet implemented"),
                };
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
        4 => dec04::run(puzzle),
        5 => dec05::run(puzzle),
        6 => dec06::run(puzzle),
        7 => dec07::run(puzzle),
        8 => dec08::run(puzzle),
        9 => dec09::run(puzzle),
        10 => dec10::run(puzzle),
        _ => return None,
    });
    let elapsed = start.elapsed().unwrap_or(Duration::ZERO);
    output.map(|result| result.map(|output| (output, elapsed)))
}

pub mod dec10 {
    use std::collections::HashSet;

    use crate::{load_input, Puzzle, Result};

    pub(crate) fn run(puzzle: Puzzle) -> Result<String> {
        match puzzle {
            Puzzle::First => first(),
            Puzzle::Second => second(),
        }
    }

    fn first() -> Result<String> {
        let input = load_input(10)?;
        let grid = Grid::new(&input);
        let path = grid.get_path();
        let max_distance = (path.len() - 1).div_ceil(2).to_string();
        Ok(max_distance)
    }

    fn second() -> Result<String> {
        let input = load_input(10)?;
        let grid = Grid::new(&input);
        let contained = grid.enclosed().to_string();
        Ok(contained)
    }

    #[derive(Debug, Copy, Clone, PartialEq)]
    enum Direction {
        North,
        South,
        East,
        West,
    }

    impl Direction {
        fn can_enter(&self, c: char) -> bool {
            c == 'S'
                || match self {
                    Self::South => ['|', 'L', 'J'],
                    Self::North => ['|', 'F', '7'],
                    Self::West => ['-', 'L', 'F'],
                    Self::East => ['-', 'J', '7'],
                }
                .contains(&c)
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    struct Position(usize, usize, usize, usize);

    impl Position {
        fn step(&self, dir: Direction) -> Option<Position> {
            match dir {
                Direction::North if self.0 > 0 => {
                    Some(Position(self.0 - 1, self.1, self.2, self.3))
                }
                Direction::South if self.0 < self.2 => {
                    Some(Position(self.0 + 1, self.1, self.2, self.3))
                }
                Direction::West if self.1 > 0 => Some(Position(self.0, self.1 - 1, self.2, self.3)),
                Direction::East if self.1 < self.3 => {
                    Some(Position(self.0, self.1 + 1, self.2, self.3))
                }
                _ => None,
            }
        }
    }

    struct Grid<'a> {
        rows: Vec<&'a str>,
        height: usize,
        width: usize,
    }

    impl<'a> Grid<'a> {
        fn new(input: &'a str) -> Self {
            let rows = input.lines().collect::<Vec<_>>();
            let height = rows.len();
            let width = rows[0].len();
            Self {
                rows,
                height,
                width,
            }
        }

        fn get(&self, pos: Position) -> char {
            self.rows[pos.0]
                .chars()
                .nth(pos.1)
                .expect("position is in bounds")
        }

        fn find_start(&self) -> Position {
            self.rows
                .iter()
                .enumerate()
                .find_map(|(row, line)| {
                    line.chars()
                        .position(|c| c == 'S')
                        .map(|col| Position(row, col, self.width - 1, self.height - 1))
                })
                .expect("Grid must contain 'S'")
        }

        fn get_path(&self) -> HashSet<Position> {
            use Direction::*;
            let mut path = HashSet::new();
            let start = self.find_start();
            path.insert(start);
            let mut curr = start;
            'outer: loop {
                let curr_char = self.get(curr);
                let dirs = match curr_char {
                    'S' => &[North, South, East, West][..],
                    '|' => &[North, South][..],
                    '-' => &[East, West][..],
                    'L' => &[North, East][..],
                    'J' => &[North, West][..],
                    'F' => &[South, East][..],
                    '7' => &[South, West][..],
                    c => unreachable!("Error at position {curr:?}: {c} can't be in path"),
                };
                for direction in dirs {
                    if let Some(next) = curr.step(*direction) {
                        let next_char = self.get(next);
                        let can_enter = direction.can_enter(next_char);
                        if can_enter && !path.contains(&next) {
                            path.insert(next);
                            curr = next;
                            break;
                        }
                        if path.len() > 2 && next_char == 'S' {
                            break 'outer;
                        }
                    }
                }
            }
            path
        }

        fn enclosed(&self) -> usize {
            use Direction::*;
            let path = self.get_path();
            let mut contained_count = 0;
            for row in 0..self.width {
                let mut contained = false;
                let mut from_dir = None;
                for col in 0..self.height {
                    let pos = Position(row, col, self.width - 1, self.height - 1);
                    let part_of_path = path.contains(&pos);
                    let curr = self.get(pos);
                    if part_of_path {
                        match curr {
                            '-' => continue,
                            '|' => contained = !contained,
                            'F' => from_dir = Some(South),
                            'L' => from_dir = Some(North),
                            'J' | '7' => match (from_dir, curr) {
                                (Some(South), 'J') | (Some(North), '7') => contained = !contained,
                                _ => continue,
                            },
                            'S' => {
                                let exit_north = North.can_enter(
                                    self.get(Position(pos.0.saturating_sub(1), pos.1, pos.2, pos.3))
                                );
                                let exit_south = South.can_enter(
                                    self.get(Position(pos.0 + 1, pos.1, pos.2, pos.3))
                                );
                                match (from_dir, exit_north, exit_south) {
                                    (Some(North), _, true) => contained = !contained,
                                    (Some(South), true, _) => contained = !contained,
                                    (None, _, _) => contained = !contained,
                                    _ => continue,
                                }
                            }
                            x => unreachable!("Char {x} at ({row}, {col}) can't be in path"),
                        }
                    } else if contained {
                        contained_count += 1;
                    }
                }
            }
            contained_count
        }
    }
}

pub mod dec09 {
    use crate::{load_input, parse_as, Puzzle, Result};

    pub(crate) fn run(puzzle: Puzzle) -> Result<String> {
        match puzzle {
            Puzzle::First => first(),
            Puzzle::Second => second(),
        }
    }

    fn first() -> Result<String> {
        let sum_next = parse_input()?
            .into_iter()
            .map(Value::into_next)
            .sum::<i64>()
            .to_string();
        Ok(sum_next)
    }

    fn second() -> Result<String> {
        let sum_prev = parse_input()?
            .into_iter()
            .map(Value::into_prev)
            .sum::<i64>()
            .to_string();
        Ok(sum_prev)
    }

    struct Value {
        history: Vec<i64>,
    }

    impl Value {
        fn new(history: Vec<i64>) -> Self {
            Self { history }
        }

        fn into_next(mut self) -> i64 {
            Self::find_next(self.history.as_mut_slice())
        }

        fn find_next(history: &mut [i64]) -> i64 {
            let mut end_idx = history.len() - 1;
            while history.iter().take(end_idx + 1).any(|v| *v != 0) {
                for idx in 0..end_idx {
                    let diff = history[idx + 1] - history[idx];
                    history[idx] = diff;
                }
                end_idx -= 1;
            }
            history.iter().sum::<i64>()
        }

        fn into_prev(mut self) -> i64 {
            Self::find_prev(self.history.as_mut_slice())
        }

        fn find_prev(history: &mut [i64]) -> i64 {
            let mut start_idx = 1;
            while history.iter().skip(start_idx).any(|v| *v != 0) {
                for idx in (start_idx..history.len()).rev() {
                    let diff = history[idx] - history[idx - 1];
                    history[idx] = diff;
                }
                start_idx += 1;
            }
            history
                .iter()
                .take(start_idx + 1)
                .rev()
                .fold(0, |prev, x| x - prev)
        }
    }

    fn parse_input() -> Result<Vec<Value>> {
        load_input(9)?
            .lines()
            .map(|line| {
                let history = line
                    .split_whitespace()
                    .map(parse_as::<i64>)
                    .collect::<Result<Vec<i64>>>()?;
                Ok(Value::new(history))
            })
            .collect::<Result<Vec<Value>>>()
    }
}

pub mod dec08 {
    use crate::{load_input, Puzzle, Result};

    pub(crate) fn run(puzzle: Puzzle) -> Result<String> {
        match puzzle {
            Puzzle::First => first(),
            Puzzle::Second => second(),
        }
    }

    fn first() -> Result<String> {
        let input = load_input(8)?;
        let len = Network::new(&input)?
            .path_len(|n| n.id == "AAA", |n| n.id == "ZZZ")?
            .to_string();
        Ok(len)
    }

    fn second() -> Result<String> {
        let input = load_input(8)?;
        let len = Network::new(&input)?
            .path_len(|n| n.id.ends_with('A'), |n| n.id.ends_with('Z'))?
            .to_string();
        Ok(len)
    }

    struct Network<'a> {
        nodes: Vec<Node<'a>>,
        directions: Directions<'a>,
    }

    impl<'a> Network<'a> {
        fn new(input: &'a str) -> Result<Self> {
            let (directions, rest) = input.split_once("\n\n").ok_or("Missing '\\n\\n'")?;
            let directions = Directions::new(directions)?;
            let mut nodes = rest
                .lines()
                .map(Node::from_line)
                .collect::<Result<Vec<Node>>>()?;
            for idx in 0..nodes.len() {
                let mut source = nodes[idx];
                source.left = match source.left {
                    Neighbor::Unresolved(l) => Neighbor::Resolved(
                        nodes
                            .iter()
                            .position(|n| n.id == l)
                            .ok_or(format!("Can't find node for left item {l}"))?,
                    ),
                    left @ Neighbor::Resolved(_) => left,
                };
                source.right = match source.right {
                    Neighbor::Unresolved(r) => Neighbor::Resolved(
                        nodes
                            .iter()
                            .position(|n| n.id == r)
                            .ok_or(format!("Can't find node for right item {r}"))?,
                    ),
                    right @ Neighbor::Resolved(_) => right,
                };
                nodes[idx] = source;
            }
            Ok(Self { nodes, directions })
        }

        fn path_len<S, E>(&mut self, starts: S, ends: E) -> Result<usize>
        where
            S: Fn(&Node) -> bool,
            E: Fn(&Node) -> bool,
        {
            let combined_len = self
                .nodes
                .iter()
                .enumerate()
                .filter(|(_, n)| starts(n))
                .map(|(mut idx, _)| {
                    let mut steps = 0;
                    for direction in self.directions.cycle() {
                        steps += 1;
                        let next = match direction {
                            Direction::Left => self.nodes[idx].left,
                            Direction::Right => self.nodes[idx].right,
                        };
                        idx = match next {
                            Neighbor::Resolved(i) => i,
                            Neighbor::Unresolved(_) => {
                                unreachable!("Neighbor {next:?} is not resolved")
                            }
                        };
                        if ends(&self.nodes[idx]) {
                            return steps;
                        }
                    }
                    unreachable!("Loop never ends")
                })
                .reduce(lcm)
                .expect("There is at least one path");
            Ok(combined_len)
        }
    }

    // Find LCM using Stein's algorithm for GCD
    fn lcm(mut a: usize, mut b: usize) -> usize {
        let mult = a * b;
        if a == 0 || b == 0 {
            return a + b;
        }
        let trailing_pows_two = (a | b).trailing_zeros();
        a >>= trailing_pows_two;
        b >>= trailing_pows_two;
        while a != b {
            if a < b {
                std::mem::swap(&mut a, &mut b);
            }
            a -= b;
            a >>= a.trailing_zeros();
        }
        mult / (a << trailing_pows_two)
    }

    #[derive(Debug, Clone, Copy)]
    enum Neighbor<'a> {
        Unresolved(&'a str),
        Resolved(usize),
    }

    #[derive(Debug, Clone, Copy)]
    struct Node<'a> {
        id: &'a str,
        left: Neighbor<'a>,
        right: Neighbor<'a>,
    }

    impl<'a> Node<'a> {
        fn from_line(line: &'a str) -> Result<Self> {
            let (me, others) = line.split_once(" = (").ok_or("No ' = ('")?;
            let (left, right) = others.split_once(", ").ok_or("No ', '")?;
            let right = right.strip_suffix(')').ok_or("No ')'")?;
            Ok(Self {
                id: me,
                left: Neighbor::Unresolved(left),
                right: Neighbor::Unresolved(right),
            })
        }
    }

    #[derive(Clone, Copy)]
    struct Directions<'a> {
        instructions: &'a str,
        index: usize,
    }

    impl<'a> Directions<'a> {
        fn new(instructions: &'a str) -> Result<Self> {
            match instructions.chars().find(|c| c != &'L' && c != &'R') {
                Some(c) => Err(format!("Invalid path character '{c}'").into()),
                None => Ok(Self {
                    instructions,
                    index: 0,
                }),
            }
        }
    }

    impl<'a> Iterator for Directions<'a> {
        type Item = Direction;
        fn next(&mut self) -> Option<Self::Item> {
            self.instructions.chars().nth(self.index).map(|c| {
                self.index += 1;
                match c {
                    'L' => Direction::Left,
                    'R' => Direction::Right,
                    _ => unreachable!(),
                }
            })
        }
    }

    enum Direction {
        Left,
        Right,
    }
}

pub mod dec07 {
    use std::cmp::Ordering;

    use crate::{load_input, parse_as, Puzzle, Result};

    pub(crate) fn run(puzzle: Puzzle) -> Result<String> {
        match puzzle {
            Puzzle::First => first(),
            Puzzle::Second => second(),
        }
    }

    fn first() -> Result<String> {
        let mut hands = parse_input(Puzzle::First)?;
        hands.sort_unstable();
        let total_winnings = hands
            .iter()
            .enumerate()
            .map(|(rank, hand)| hand.winnings(rank + 1))
            .sum::<usize>()
            .to_string();
        Ok(total_winnings)
    }

    fn second() -> Result<String> {
        let mut hands = parse_input(Puzzle::Second)?;
        hands.sort_unstable();
        let total_winnings = hands
            .iter()
            .enumerate()
            .map(|(rank, hand)| hand.winnings(rank + 1))
            .sum::<usize>()
            .to_string();
        Ok(total_winnings)
    }

    #[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
    enum Type {
        HighCard,
        OnePair,
        TwoPair,
        ThreeOfAKind,
        FullHouse,
        FourOfAKind,
        FiveOfAKind,
    }

    impl Type {
        fn from_cards(val: &[u8; 5], jokers: bool) -> Self {
            let mut counts = [0u8; 13];
            val.iter().for_each(|card| counts[*card as usize] += 1);
            let joker_count = if jokers {
                let tmp = counts[0];
                counts[0] = 0;
                tmp
            } else {
                0
            };
            let fives = counts.iter().filter(|c| c == &&5).count();
            let fours = counts.iter().filter(|c| c == &&4).count();
            let threes = counts.iter().filter(|c| c == &&3).count();
            let twos = counts.iter().filter(|c| c == &&2).count();
            match (fives, fours, threes, twos) {
                (1, _, _, _) => Self::FiveOfAKind,
                (_, 1, _, _) => match joker_count {
                    1 => Self::FiveOfAKind,
                    _ => Self::FourOfAKind,
                },
                (_, _, 1, 1) => Self::FullHouse,
                (_, _, 1, 0) => match joker_count {
                    2 => Self::FiveOfAKind,
                    1 => Self::FourOfAKind,
                    _ => Self::ThreeOfAKind,
                },
                (_, _, _, 2) => match joker_count {
                    1 => Self::FullHouse,
                    _ => Self::TwoPair,
                },
                (_, _, _, 1) => match joker_count {
                    3 => Self::FiveOfAKind,
                    2 => Self::FourOfAKind,
                    1 => Self::ThreeOfAKind,
                    _ => Self::OnePair,
                },
                _ => match joker_count {
                    5 | 4 => Self::FiveOfAKind,
                    3 => Self::FourOfAKind,
                    2 => Self::ThreeOfAKind,
                    1 => Self::OnePair,
                    _ => Self::HighCard,
                },
            }
        }
    }

    #[derive(Debug, PartialEq, Eq)]
    struct Hand {
        hand_type: Type,
        cards: [u8; 5],
        bet: usize,
    }

    impl Hand {
        fn parse(line: &str, jokers: bool) -> Result<Hand> {
            let (hand, bet) = line
                .split_once(' ')
                .ok_or(format!("{line} is missing space"))?;
            let bet = parse_as::<usize>(bet)?;
            debug_assert!(hand.len() == 5, "{hand} doesn't have 5 characters");

            let mut cards = [0u8; 5];
            let chars = hand.chars().enumerate();
            let offset = if jokers { 1 } else { 0 };
            for (idx, card) in chars {
                match card {
                    '2' => cards[idx] = offset,
                    '3' => cards[idx] = 1 + offset,
                    '4' => cards[idx] = 2 + offset,
                    '5' => cards[idx] = 3 + offset,
                    '6' => cards[idx] = 4 + offset,
                    '7' => cards[idx] = 5 + offset,
                    '8' => cards[idx] = 6 + offset,
                    '9' => cards[idx] = 7 + offset,
                    'T' => cards[idx] = 8 + offset,
                    'J' => cards[idx] = if jokers { 0 } else { 9 },
                    'Q' => cards[idx] = 10,
                    'K' => cards[idx] = 11,
                    'A' => cards[idx] = 12,
                    x => Err(format!("{x} is not a valid card"))?,
                }
            }
            let hand_type = Type::from_cards(&cards, jokers);
            Ok(Self {
                hand_type,
                cards,
                bet,
            })
        }

        fn winnings(&self, rank: usize) -> usize {
            self.bet * rank
        }
    }

    impl PartialOrd for Hand {
        fn partial_cmp(&self, rhs: &Self) -> Option<Ordering> {
            Some(self.cmp(rhs))
        }
    }

    impl Ord for Hand {
        fn cmp(&self, rhs: &Self) -> Ordering {
            self.hand_type.cmp(&rhs.hand_type).then_with(|| {
                self.cards
                    .iter()
                    .zip(rhs.cards)
                    .find(|(c, rhs_c)| c.cmp(&rhs_c).is_ne())
                    .map(|(c, rhs_c)| c.cmp(&rhs_c))
                    .unwrap_or(Ordering::Equal)
            })
        }
    }

    fn parse_input(part: Puzzle) -> Result<Vec<Hand>> {
        let jokers = matches!(part, Puzzle::Second);
        load_input(7)?
            .lines()
            .map(|line| Hand::parse(line, jokers))
            .collect()
    }
}

pub mod dec06 {
    use std::cmp::Ordering;

    use crate::{load_input, parse_as, Puzzle, Result};

    pub(crate) fn run(puzzle: Puzzle) -> Result<String> {
        match puzzle {
            Puzzle::First => first(),
            Puzzle::Second => second(),
        }
    }

    fn first() -> Result<String> {
        let races = parse_input(Puzzle::First)?;
        let total = races
            .into_iter()
            .map(|race| race.num_solutions())
            .product::<usize>()
            .to_string();
        Ok(total)
    }

    fn second() -> Result<String> {
        let race = parse_input(Puzzle::Second)?.remove(0);
        Ok(race.num_solutions().to_string())
    }

    #[derive(Debug)]
    struct Race {
        time: usize,
        record: usize,
    }

    impl Race {
        fn num_solutions(&self) -> usize {
            let min = binary_search(0, self.time / 2 - 1, |val| {
                match (self.is_solution(val), self.is_solution(val + 1)) {
                    (true, true) => Ordering::Greater,
                    (false, false) => Ordering::Less,
                    (false, true) => Ordering::Equal,
                    (true, false) => unreachable!(),
                }
            });
            let max = binary_search(self.time / 2, self.time, |val| {
                match (self.is_solution(val), self.is_solution(val + 1)) {
                    (true, true) => Ordering::Less,
                    (false, false) => Ordering::Greater,
                    (true, false) => Ordering::Equal,
                    (false, true) => unreachable!(),
                }
            });
            let (max, min) = (max.expect("Max exists"), min.expect("Min exists"));
            max - min
        }

        fn is_solution(&self, val: usize) -> bool {
            val * (self.time - val) > self.record
        }
    }

    impl From<(usize, usize)> for Race {
        fn from(val: (usize, usize)) -> Self {
            Self {
                time: val.0,
                record: val.1,
            }
        }
    }

    #[inline]
    fn binary_search<F>(mut bottom: usize, mut top: usize, cond: F) -> Option<usize>
    where
        F: Fn(usize) -> Ordering,
    {
        while bottom < top {
            if top - bottom < 10 {
                return (bottom..=top).find(|i| cond(*i) == Ordering::Equal);
            } else {
                let mid = (top + bottom) / 2;
                match cond(mid) {
                    Ordering::Less => bottom = mid - 1,
                    Ordering::Greater => top = mid + 1,
                    Ordering::Equal => return Some(mid),
                }
            }
        }
        None
    }

    fn parse_input(part: Puzzle) -> Result<Vec<Race>> {
        let input = load_input(6)?;
        let mut lines = input.lines();
        let times = lines
            .next()
            .and_then(|l| l.strip_prefix("Time:"))
            .ok_or("No line 1")?;
        let records = lines
            .next()
            .and_then(|l| l.strip_prefix("Distance:"))
            .ok_or("No line 2")?;
        match part {
            Puzzle::First => {
                let times = times
                    .split_whitespace()
                    .filter(|n| !n.is_empty())
                    .map(parse_as::<usize>)
                    .collect::<Result<Vec<usize>>>()?;
                let records = records
                    .split_whitespace()
                    .filter(|n| !n.is_empty())
                    .map(parse_as::<usize>)
                    .collect::<Result<Vec<usize>>>()?;
                Ok(times.into_iter().zip(records).map(Race::from).collect())
            }
            Puzzle::Second => {
                let time = parse_as::<usize>(&times.replace(' ', ""))?;
                let record = parse_as::<usize>(&records.replace(' ', ""))?;
                Ok(vec![(time, record).into()])
            }
        }
    }
}

pub mod dec05 {
    use crate::{load_input, parse_as, Puzzle, Result};

    pub(crate) fn run(puzzle: Puzzle) -> Result<String> {
        match puzzle {
            Puzzle::First => first(),
            Puzzle::Second => second(),
        }
    }

    fn first() -> Result<String> {
        let (seeds, ranges) = parse_input()?;
        let min_seed_location = seeds
            .into_iter()
            .map(|seed| {
                ranges
                    .iter()
                    .find_map(|range| range.translate(seed))
                    .unwrap_or(seed)
            })
            .min()
            .ok_or("There is at least one seed")?
            .to_string();
        Ok(min_seed_location)
    }

    fn second() -> Result<String> {
        let (seeds, ranges) = parse_input()?;
        let mut seed_ranges = Vec::new();
        for x in (1..seeds.len()).step_by(2) {
            seed_ranges.push((seeds[x - 1], seeds[x - 1] + seeds[x] + 1));
        }

        let min_seed_location = seed_ranges
            .into_iter()
            .map(|seed_range| {
                ranges
                    .iter()
                    .filter_map(|range| range.translate_min(seed_range))
                    .min()
                    .unwrap_or(seed_range.0)
            })
            .min()
            .ok_or("There is at least one seed range")?
            .to_string();
        Ok(min_seed_location)
    }

    #[derive(Debug, Clone, Copy)]
    struct Range {
        input_start: usize,
        input_end: usize,
        output_start: usize,
        output_end: usize,
    }

    enum CombinedRange {
        Input(Range),
        Output(Range),
        Overlap(Range),
        None,
    }

    impl Range {
        fn new(
            input_start: usize,
            input_end: usize,
            output_start: usize,
            output_end: usize,
        ) -> Self {
            Self {
                input_start,
                input_end,
                output_start,
                output_end,
            }
        }

        fn translate(&self, val: usize) -> Option<usize> {
            if val >= self.input_start && val <= self.input_end {
                if self.input_start > self.output_start {
                    Some(val - (self.input_start - self.output_start))
                } else {
                    Some(val + (self.output_start - self.input_start))
                }
            } else {
                None
            }
        }

        fn translate_min(&self, range: (usize, usize)) -> Option<usize> {
            if range.1 < self.input_start || range.0 > self.input_end {
                None
            } else {
                self.translate(std::cmp::max(range.0, self.input_start))
            }
        }

        fn overlaps(input: &Range, output: &Range) -> bool {
            input.output_end >= output.input_start && input.output_start <= output.input_end
        }

        fn input_for(&self, output: usize) -> usize {
            let offset = output - self.output_start;
            self.input_start + offset
        }

        fn output_for(&self, input: usize) -> usize {
            let offset = input - self.input_start;
            self.output_start + offset
        }

        // Splits an input range based on a value in its output range.
        // The provided value is part of the SECOND range returned, if applicable.
        fn split_at_output(self, second_start: usize) -> (Option<Range>, Option<Range>) {
            if second_start <= self.output_start {
                (None, Some(self))
            } else if second_start > self.output_end {
                (Some(self), None)
            } else {
                let first_len = second_start - 1 - self.output_start;
                let first = Range::new(
                    self.input_start,
                    self.input_start + first_len,
                    self.output_start,
                    self.output_start + first_len,
                );
                let second = Range::new(
                    first.input_end + 1,
                    self.input_end,
                    first.output_end + 1,
                    self.output_end,
                );
                (Some(first), Some(second))
            }
        }

        fn combine(input: Range, output: Range) -> [CombinedRange; 3] {
            use std::cmp::Ordering;
            use CombinedRange::*;
            if !Range::overlaps(&input, &output) {
                [Input(input), Output(output), None]
            } else {
                let (low, high) = (
                    std::cmp::max(input.output_start, output.input_start),
                    std::cmp::min(input.output_end, output.input_end),
                );
                let overlap = Overlap(Range::new(
                    input.input_for(low),
                    input.input_for(high),
                    output.output_for(low),
                    output.output_for(high),
                ));
                match input.output_start.cmp(&output.input_start) {
                    Ordering::Less => {
                        let first = Input(Range::new(
                            input.input_start,
                            input.input_for(low - 1),
                            input.output_start,
                            low - 1,
                        ));
                        match input.output_end.cmp(&output.input_end) {
                            Ordering::Less => [
                                first,
                                overlap,
                                Output(Range::new(
                                    high + 1,
                                    output.input_end,
                                    output.output_for(high + 1),
                                    output.output_end,
                                )),
                            ],
                            Ordering::Equal => [first, overlap, None],
                            Ordering::Greater => [
                                first,
                                overlap,
                                Input(Range::new(
                                    input.input_for(high + 1),
                                    input.input_end,
                                    high + 1,
                                    input.output_end,
                                )),
                            ],
                        }
                    }
                    Ordering::Equal => match input.output_end.cmp(&output.input_end) {
                        Ordering::Less => [
                            overlap,
                            Output(Range::new(
                                high + 1,
                                output.input_end,
                                output.output_for(high + 1),
                                output.output_end,
                            )),
                            None,
                        ],
                        Ordering::Equal => [overlap, None, None],
                        Ordering::Greater => [
                            overlap,
                            Input(Range::new(
                                input.input_for(high + 1),
                                input.input_end,
                                high + 1,
                                input.output_end,
                            )),
                            None,
                        ],
                    },
                    Ordering::Greater => {
                        let first = Output(Range::new(
                            output.input_start,
                            low - 1,
                            output.output_start,
                            output.output_for(low - 1),
                        ));
                        match input.output_end.cmp(&output.input_end) {
                            Ordering::Less => [
                                first,
                                overlap,
                                Output(Range::new(
                                    high + 1,
                                    output.input_end,
                                    output.output_for(high + 1),
                                    output.output_end,
                                )),
                            ],
                            Ordering::Equal => [first, overlap, None],
                            Ordering::Greater => [
                                first,
                                overlap,
                                Input(Range::new(
                                    input.input_for(high + 1),
                                    input.input_end,
                                    high + 1,
                                    input.output_end,
                                )),
                            ],
                        }
                    }
                }
            }
        }
    }

    fn parse_input() -> Result<(Vec<usize>, Vec<Range>)> {
        let input = load_input(5)?;
        let (seeds, maps) = input.split_once("\n\n").ok_or("No section divider")?;
        let seeds = seeds
            .strip_prefix("seeds: ")
            .ok_or("No seeds prefix")?
            .split(' ')
            .map(parse_as::<usize>)
            .collect::<Result<Vec<usize>>>()?;
        let map = maps.split("\n\n").try_fold(Vec::new(), |prior, section| {
            let mut ranges = section
                .split('\n')
                .skip(1)
                .filter(|line| !line.is_empty())
                .map(|line| {
                    let nums = line
                        .split(' ')
                        .map(parse_as::<usize>)
                        .collect::<Result<Vec<usize>>>()?;
                    if let [dest, source, len] = nums.as_slice() {
                        Ok(Range::new(*source, source + len - 1, *dest, dest + len - 1))
                    } else {
                        Err(format!("Wrong number of elements on line {line}").into())
                    }
                })
                .collect::<Result<Vec<Range>>>()?;
            let mut new_ranges = Vec::new();
            for mut input in prior {
                let mut outputs = ranges
                    .iter()
                    .enumerate()
                    .filter(|(_, output)| Range::overlaps(&input, output))
                    .map(|(idx, _)| idx)
                    .collect::<Vec<_>>();
                outputs.sort_unstable();
                let mut outputs = outputs
                    .into_iter()
                    .rev()
                    .map(|idx| ranges.remove(idx))
                    .collect::<Vec<_>>();
                outputs.sort_unstable_by_key(|range| range.input_start);
                if outputs.len() > 1 {
                    for output in outputs {
                        match input.split_at_output(output.input_end + 1) {
                            (Some(overlaps), Some(after)) => {
                                input = after;
                                for range in Range::combine(overlaps, output) {
                                    match range {
                                        CombinedRange::Input(x) | CombinedRange::Overlap(x) => {
                                            new_ranges.push(x)
                                        }
                                        CombinedRange::Output(x) => ranges.push(x),
                                        CombinedRange::None => (),
                                    }
                                }
                            }
                            (Some(overlaps), None) => {
                                for range in Range::combine(overlaps, output) {
                                    match range {
                                        CombinedRange::Input(x) | CombinedRange::Overlap(x) => {
                                            new_ranges.push(x)
                                        }
                                        CombinedRange::Output(x) => ranges.push(x),
                                        CombinedRange::None => (),
                                    }
                                }
                            }
                            (None, Some(_)) => unreachable!(),
                            (None, None) => unreachable!(),
                        }
                    }
                } else if !outputs.is_empty() {
                    let output = outputs[0];
                    for range in Range::combine(input, output) {
                        match range {
                            CombinedRange::Input(x) | CombinedRange::Overlap(x) => {
                                new_ranges.push(x)
                            }
                            CombinedRange::Output(x) => ranges.push(x),
                            CombinedRange::None => (),
                        }
                    }
                } else {
                    new_ranges.push(input);
                }
            }
            new_ranges.extend(ranges);
            let output: Result<Vec<_>> = Ok(new_ranges);
            output
        })?;

        Ok((seeds, map))
    }
}

pub mod dec04 {
    use std::convert::TryFrom;

    use crate::{load_input, Puzzle, Result};

    pub(crate) fn run(puzzle: Puzzle) -> Result<String> {
        match puzzle {
            Puzzle::First => first(),
            Puzzle::Second => second(),
        }
    }

    fn first() -> Result<String> {
        let summed_scores = load_input(4)?
            .lines()
            .map(Card::try_from)
            .try_fold(0, |acc, x| x.map(|card| acc + card.score()))?
            .to_string();
        Ok(summed_scores)
    }

    fn second() -> Result<String> {
        let cards = load_input(4)?
            .lines()
            .map(Card::try_from)
            .collect::<Result<Vec<Card>>>()?;
        let num_cards = cards.len();
        let mut copies = vec![1; num_cards];
        for (idx, card) in cards.iter().enumerate() {
            let curr_count = copies[idx];
            let num_to_boost = card.num_winners();
            for item in copies
                .iter_mut()
                .take((std::cmp::min(idx + num_to_boost, num_cards)) + 1)
                .skip(idx + 1)
            {
                *item += curr_count;
            }
        }
        let total_copies = copies.iter().sum::<u32>().to_string();
        Ok(total_copies)
    }

    #[derive(Debug)]
    struct Card {
        winners: usize,
    }

    impl Card {
        fn num_winners(&self) -> usize {
            self.winners
        }

        fn score(&self) -> u32 {
            1 << self.winners >> 1
        }
    }

    impl TryFrom<&str> for Card {
        type Error = Box<dyn std::error::Error>;
        fn try_from(value: &str) -> Result<Self> {
            type Res<T> = std::result::Result<Vec<T>, std::num::ParseIntError>;

            let (_, data) = value.split_once(':').ok_or("Line is missing card id")?;
            let (cards, have) = data.split_once('|').ok_or("Line is missing '|'")?;
            let cards: Vec<_> = cards
                .trim()
                .split(' ')
                .filter(|s| !s.is_empty())
                .map(|n| n.trim().parse::<u32>())
                .collect::<Res<_>>()?;
            let winners = have
                .trim()
                .split(' ')
                .filter(|s| !s.is_empty())
                .map(|n| {
                    n.trim()
                        .parse::<u32>()
                        .map(|val| cards.iter().find(|v| **v == val))
                })
                .try_fold(0, |acc, x| x.map(|opt| acc + opt.map(|_| 1).unwrap_or(0)))?;
            Ok(Card { winners })
        }
    }
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

fn parse_as<T>(val: &str) -> Result<T>
where
    T: std::str::FromStr<Err = std::num::ParseIntError>,
{
    Ok(val.parse::<T>().map_err(|e| {
        format!(
            "Could not parse {val} as {}: {e}",
            std::any::type_name::<T>()
        )
    })?)
}

#[cfg(test)]
mod tests {
    use crate::{run_puzzle, Puzzle, Result};

    #[test]
    fn check_correctness() -> Result<()> {
        let answers = [
            "54159", // Day 1
            "53866",
            "2317", // Day 2
            "74804",
            "525119", // Day 3
            "76504829",
            "25231", // Day 4
            "9721255",
            "51580674", // Day 5
            "99751240",
            "1195150", // Day 6
            "42550411",
            "250058342", // Day 7
            "250506580",
            "15989", // Day 8
            "13830919117339",
            "1798691765", // Day 9
            "1104",
            "6870", // Day 10
            "287",
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
