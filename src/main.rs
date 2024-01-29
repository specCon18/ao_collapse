use std::collections::HashMap;
use rand::{distributions::{Distribution, WeightedIndex}, rngs::ThreadRng, thread_rng};

/// Represents a tile in the grid.
type Tile = i32;

/// A collection of possible states a cell can be in.
type PossibleStates = Vec<Tile>;

/// A collection of positions relative to the current cell
enum ConstraintRelativePos {
    Top,
    Bottom,
    Left,
    Right,
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight
}

enum ConstraintRule {
    IsEqual,
    IsNotEqual
}

struct Constraint {
    relative_pos: ConstraintRelativePos,
    rule: ConstraintRule,
    current_tile: Tile,
    adjacent_tiles: PossibleStates
}


/// Represents a cell in the grid.
#[derive(Clone)]
struct Cell {
    possible_states: PossibleStates,
    entropy: f64,
    collapsed: bool,
}

impl Cell {
    /// Creates a new cell with given possible states.
    ///
    /// # Arguments
    ///
    /// * `possible_states` - A vector of possible states the cell can take.
    ///
    /// # Returns
    ///
    /// A new `Cell` instance.
    fn new(possible_states: PossibleStates) -> Self {
        let entropy = calculate_entropy(&possible_states);
        Cell {
            possible_states,
            entropy,
            collapsed: false,
        }
    }
}

/// Information about a cell, specifically for use in a min-heap.
#[derive(Clone)]
struct CellInfo {
    entropy: f64,
    coordinates: (usize, usize),
}

/// A min-heap data structure for managing cells based on their entropy.
struct MinHeap {
    data: Vec<CellInfo>,
}

impl MinHeap {
    /// Creates a new, empty `MinHeap`.
    fn new() -> Self {
        MinHeap { data: Vec::new() }
    }

    /// Adds a `CellInfo` element to the heap.
    ///
    /// # Arguments
    ///
    /// * `element` - The `CellInfo` to be added.
    fn add(&mut self, element: CellInfo) {
        self.data.push(element);
        self.heapify_up(self.data.len() - 1);
    }

    /// Removes and returns the smallest `CellInfo` from the heap.
    ///
    /// # Returns
    ///
    /// The smallest `CellInfo`, if the heap is not empty.
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

    /// Maintains the heap property by moving an element up.
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the element to be moved up.
    fn heapify_up(&mut self, mut index: usize) {
        while index != 0 {
            let parent_index = (index - 1) / 2;
            if self.data[index].entropy < self.data[parent_index].entropy {
                self.data.swap(parent_index, index);
            }
            index = parent_index;
        }
    }

    /// Maintains the heap property by moving an element down.
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the element to be moved down.
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

/// Calculates the Shannon entropy of a set of tiles.
///
/// # Arguments
///
/// * `data` - A slice of `Tile` instances.
///
/// # Returns
///
/// The calculated entropy as a `f64`.
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

/// Represents the entire grid of cells.
struct Grid {
    cells: Vec<Vec<Cell>>,
}

impl Grid {
    /// Creates a new grid with specified dimensions and possible states for each cell.
    ///
    /// # Arguments
    ///
    /// * `width` - The width of the grid.
    /// * `height` - The height of the grid.
    /// * `possible_states` - Possible states for each cell.
    ///
    /// # Returns
    ///
    /// A new `Grid` instance.
    fn new(width: usize, height: usize, possible_states: PossibleStates) -> Self {
        let cells = vec![vec![Cell::new(possible_states.clone()); width]; height];
        Grid { cells }
    }

    /// Observes the grid and collapses cells based on their states and neighbors.
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
    

    /// Finds the cell with minimum entropy.
    ///
    /// # Returns
    ///
    /// The coordinates of the cell with the minimum entropy.
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

    /// Collapses a cell to a single state.
    ///
    /// # Arguments
    ///
    /// * `x` - The x-coordinate of the cell.
    /// * `y` - The y-coordinate of the cell.
    /// * `rng` - A mutable reference to a random number generator.
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
    

    
    /// Updates the possible states of a cell based on its neighbors.
    ///
    /// # Arguments
    ///
    /// * `x` - The x-coordinate of the cell.
    /// * `y` - The y-coordinate of the cell.
    /// * `current_states` - The current possible states of the cell.
    ///
    /// # Returns
    ///
    /// Updated possible states after considering neighbors.
    fn updated_possible_states_based_on_neighbors(&self, x: usize, y: usize, current_states: &PossibleStates) -> PossibleStates {
        let mut neighbors = Vec::new();

        // Assuming self.cells is a 2D grid of cells
        // Collect the states of neighboring cells if they are collapsed
        if y > 0 && self.cells[y - 1][x].collapsed {
            neighbors.push(self.cells[y - 1][x].possible_states[0]);
        }
        if y < self.cells.len() - 1 && self.cells[y + 1][x].collapsed {
            neighbors.push(self.cells[y + 1][x].possible_states[0]);
        }
        if x > 0 && self.cells[y][x - 1].collapsed {
            neighbors.push(self.cells[y][x - 1].possible_states[0]);
        }
        if x < self.cells[0].len() - 1 && self.cells[y][x + 1].collapsed {
            neighbors.push(self.cells[y][x + 1].possible_states[0]);
        }

        // Filter the states based on the constraints with the neighbors
        current_states.iter().cloned().filter(|&state| {
            neighbors.iter().all(|&neighbor_state| {
                is_constraint_satisfied(&state, &neighbor_state)
            })
        }).collect()
    }
    

    /// Generates the final state of the grid.
    fn generate(&mut self) {
        while self.cells.iter().any(|row| row.iter().any(|cell| !cell.collapsed)) {
            self.observe_and_collapse();
            for row in &mut self.cells {
                for cell in row {
                    if cell.possible_states.is_empty() && !cell.collapsed {
                        cell.possible_states = vec![99999];
                        cell.collapsed = true;
                    }
                }
            }
        }
    }

    /// Prints the current state of the grid.
    fn print_grid(&self) {
        for row in &self.cells {
            for cell in row {
                let state = cell.possible_states.get(0).map_or(0, |&s| s);
                print!("{:05} ", state); // Print each state as a 2-digit number with leading zeros if necessary
            }
            println!();
        }
    }

}

/// Checks if the constraint between two cells is satisfied.
///
/// # Arguments
///
/// * `cell_state` - The state of the current cell.
/// * `neighbor_state` - The state of the neighboring cell.
///
/// # Returns
///
/// `true` if the constraint is satisfied, `false` otherwise.
fn is_constraint_satisfied(cell_state: &Tile, neighbor_state: &Tile) -> bool {
    // Check if the neighbor's state is either one more or one less than the current cell's state
    *neighbor_state == *cell_state / 2 || *neighbor_state == *cell_state * 2
}
// //! Induces failure if 2 isnt in the set
// /// Checks if the constraints between two cells are satisfied.
// ///
// /// # Arguments
// ///
// /// * `cell_state` - The state of the current cell.
// /// * `neighbor_state` - The state of the neighboring cell.
// ///
// /// # Returns
// ///
// /// `true` if the constraints are satisfied, `false` otherwise.
//// fn is_constraint_satisfied(cell_state: &Tile, neighbor_state: &Tile) -> bool {
//     // Check if the neighbor's state is either one more or one less than the current cell's state
////     let basic_constraint = *neighbor_state == *cell_state + 1 || *neighbor_state == *cell_state - 1;
//     // Additional constraint: if the neighbor is 1, the current cell must be 2, and vice versa
////     let specific_constraint = (*neighbor_state == 1 && *cell_state == 2) || (*neighbor_state == 2 && *cell_state == 1);
////     basic_constraint || specific_constraint
//// }

fn main() {
    let possible_states = vec![1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768];
    let mut grid = Grid::new(16, 16, possible_states);
    
    grid.generate();
    grid.print_grid();
}
