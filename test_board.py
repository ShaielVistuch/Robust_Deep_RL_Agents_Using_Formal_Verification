from Frozen_Lake_Environment import Frozen_Lake_Environment

# Length\Width of square-shaped board
size = 3
env = Frozen_Lake_Environment(size)
env.print_environment_parameters()
env.add_layout_3_3()
env.print_environment_parameters()
env.print_on_board_current_state()