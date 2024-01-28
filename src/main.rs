use std::collections::HashMap;
use rand::{distributions::{Distribution, WeightedIndex}, rngs::ThreadRng, thread_rng};

type Tile = i32;
type PossibleStates = Vec<Tile>;

#[derive(Clone)]
struct Cell {
    possible_states: PossibleStates,
    entropy: f64,
    collapsed: bool,
}

impl Cell {
    fn new(possible_states: PossibleStates) -> Self {
        let entropy = calculate_entropy(&possible_states);
        Cell {
            possible_states,
            entropy,
            collapsed: false,
        }
    }
}

#[derive(Clone)]
struct CellInfo {
    entropy: f64,
    coordinates: (usize, usize),
}

struct MinHeap {
    data: Vec<CellInfo>,
}

impl MinHeap {
    fn new() -> Self {
        MinHeap { data: Vec::new() }
    }

    fn add(&mut self, element: CellInfo) {
        self.data.push(element);
        self.heapify_up(self.data.len() - 1);
    }

    fn pop(&mut self) -> Option<CellInfo> {
        if self.data.is_empty() {
            return None;
        }
        let last_index = self.data.len() - 1;
        self.data.swap(0, last_index);
        let result = self.data.pop();
        self.heapify_down(0);
        result
    }

    fn heapify_up(&mut self, mut index: usize) {
        while index != 0 {
            let parent_index = (index - 1) / 2;
            if self.data[index].entropy < self.data[parent_index].entropy {
                self.data.swap(parent_index, index);
            }
            index = parent_index;
        }
    }

    fn heapify_down(&mut self, mut index: usize) {
        let length = self.data.len();
        loop {
            let left_child = 2 * index + 1;
            let right_child = 2 * index + 2;

            let mut smallest = index;

            if left_child < length && self.data[left_child].entropy < self.data[smallest].entropy {
                smallest = left_child;
            }

            if right_child < length && self.data[right_child].entropy < self.data[smallest].entropy {
                smallest = right_child;
            }

            if smallest != index {
                self.data.swap(index, smallest);
                index = smallest;
            } else {
                break;
            }
        }
    }
}

fn calculate_entropy(data: &[Tile]) -> f64 {
    let mut frequency_map = HashMap::new();
    for &value in data {
        *frequency_map.entry(value).or_insert(0) += 1;
    }
    let len = data.len() as f64;
    let mut entropy = 0.0;
    for &frequency in frequency_map.values() {
        let probability = frequency as f64 / len;
        entropy -= probability * probability.log2();
    }
    entropy
}

struct Grid {
    cells: Vec<Vec<Cell>>,
}

impl Grid {
    fn new(width: usize, height: usize, possible_states: PossibleStates) -> Self {
        let cells = vec![vec![Cell::new(possible_states.clone()); width]; height];
        Grid { cells }
    }

    fn observe_and_collapse(&mut self) {
        let mut rng = thread_rng();
        
        // Collect updates in a separate step to avoid borrowing conflicts
        let mut updates = Vec::new();
        for (y, row) in self.cells.iter().enumerate() {
            for (x, cell) in row.iter().enumerate() {
                if !cell.collapsed {
                    let updated_states = self.updated_possible_states_based_on_neighbors(x, y, &cell.possible_states);
                    updates.push(((x, y), updated_states));
                }
            }
        }
    
        // Apply updates
        for ((x, y), updated_states) in updates {
            self.cells[y][x].possible_states = updated_states;
        }
    
        // Now proceed with collapse
        if let Some((x, y)) = self.find_min_entropy_cell() {
            self.collapse_cell(x, y, &mut rng);
        }
    }
    

    fn find_min_entropy_cell(&self) -> Option<(usize, usize)> {
        let mut heap = MinHeap::new();

        for (y, row) in self.cells.iter().enumerate() {
            for (x, cell) in row.iter().enumerate() {
                if !cell.collapsed {
                    heap.add(CellInfo {
                        entropy: cell.entropy,
                        coordinates: (x, y),
                    });
                }
            }
        }

        heap.pop().map(|cell_info| cell_info.coordinates)
    }

    fn collapse_cell(&mut self, x: usize, y: usize, rng: &mut ThreadRng) {
        let cell = &mut self.cells[y][x];

        if cell.possible_states.is_empty() {
            return;
        }

        let weights = cell.possible_states.iter().map(|&_state| 1).collect::<Vec<_>>();
        let dist = WeightedIndex::new(&weights).unwrap();
        let chosen_state = cell.possible_states[dist.sample(rng)];

        cell.possible_states = vec![chosen_state];
        cell.entropy = 0.0;
        cell.collapsed = true;
    }

    fn updated_possible_states_based_on_neighbors(&self, x: usize, y: usize, current_states: &PossibleStates) -> PossibleStates {
        let mut neighbors = Vec::new();
    
        if y > 0 { neighbors.push(&self.cells[y - 1][x]); }
        if y < self.cells.len() - 1 { neighbors.push(&self.cells[y + 1][x]); }
        if x > 0 { neighbors.push(&self.cells[y][x - 1]); }
        if x < self.cells[0].len() - 1 { neighbors.push(&self.cells[y][x + 1]); }
    
        let filtered_states: PossibleStates = current_states.iter()
            .filter(|&&state| {
                neighbors.iter().all(|neighbor| {
                    !neighbor.collapsed || (neighbor.collapsed && neighbor.possible_states != vec![state])
                })
            })
            .cloned()
            .collect();
    
    //    println!("Cell[{}, {}] - Before: {:?}, After: {:?}", x, y, current_states, filtered_states);
        filtered_states
    }
    


    fn generate(&mut self) {
        while self.cells.iter().any(|row| row.iter().any(|cell| !cell.collapsed)) {
            self.observe_and_collapse();

            for row in &mut self.cells {
                for cell in row {
                    if cell.possible_states.is_empty() && !cell.collapsed {
                        cell.possible_states = vec![0];
                        cell.collapsed = true;
                    }
                }
            }
        }
    }

    fn print_grid(&self) {
        for row in &self.cells {
            for cell in row {
                let state = cell.possible_states.get(0).map_or('0', |&s| {
                    if s == 0 { '0' } else { char::from_digit(s as u32, 10).unwrap_or('?') }
                });
                print!("{} ", state);
            }
            println!();
        }
    }
}

fn main() {
    let possible_states = vec![1, 2];
    let mut grid = Grid::new(16, 16, possible_states);
    
    grid.generate();
    grid.print_grid();
}
