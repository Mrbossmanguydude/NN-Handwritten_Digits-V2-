import numpy as np
import pygame
import pickle
import os

pygame.init()
pygame.display.set_icon(pygame.image.load("images\\NN_icon.png"))

WIDTH, HEIGHT = 560, 560
ROWS, COLS = 28, 28
SQUARE_SIZE = 280 // COLS
FPS = 60
SELECT_COLOR = (200, 0, 0)
DEF_BATCH_SIZE = 50
DEF_NUM_EPOCHS = 5
DEF_LEARN_RATE = 0.001
FONT = "fonts\\pixel_font-1.ttf"

class Button:
    def __init__(self, x, y, width, height, text, text_size, bordercolor=(0, 0, 0), textcolor=(0, 0, 0), thickness=5):
        self.rect = pygame.Rect(x, y, width, height)
        self.rect.topleft = (x, y)
        self.text = text
        self.text_len = len(self.text)
        self.text_size = text_size
        self.textcolor = textcolor
        self.bordercolor = bordercolor
        self.thickness = thickness
        self.clicked_ticks = 0
        self.clicked = False

    def get_clicked(self):
        #Check for a left mouse button click.
        if self.clicked_ticks >= FPS:
            mouse_pos = pygame.mouse.get_pos()
            if self.rect.collidepoint(mouse_pos):
                if pygame.mouse.get_pressed()[0]: #Index 0 specifies left mouse button.
                    self.clicked = True #Button is clicked if a collision with mouse and Rect is detected.
        else:
            self.clicked = False #Otherwise, the button is not clicked.

    def draw(self, screen):
        #Draws out the button and its border.
        draw_highlighted_rect(screen, self.rect, self.bordercolor, self.bordercolor, self.thickness, self.thickness)
        draw_text(screen, FONT, self.text, (self.rect.x + 15, self.rect.y), self.text_size, self.textcolor)

def reset_buttons(buttons : list):
    for button in buttons:
        button.clicked_ticks = 0
        button.clicked = False

def increment_button_ticks(buttons : list):
    for button in buttons:
        if button.clicked_ticks < FPS: #Uses FPS to ensure only close to a second is there for button delay.
            button.clicked_ticks += 1

class Layer:
    def __init__(self, num_inputs, num_outputs):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.weights = np.random.randn(num_inputs, num_outputs) * np.sqrt(2. / num_inputs)  # The initialization for ReLU
        self.biases = np.zeros(num_outputs)
        self.activations = []

    def forward(self, inputs):
        self.inputs = np.array(inputs)
        z = np.dot(self.inputs, self.weights) + self.biases
        if self.num_outputs == 10:  # Apply softmax for the output layer
            self.activations = self.softmax(z)
        else:
            self.activations = self.relu(z)
        return self.activations

    def backward(self, gradient_loss_output):
        gradient_loss_input = np.multiply(gradient_loss_output, self.d_relu(self.activations))

        gradient_input_weights = self.inputs
        gradient_input_inputs = self.weights

        gradient_loss_weights = np.dot(gradient_input_weights.T, gradient_loss_input)
        gradient_loss_bias = np.sum(gradient_loss_input, axis=0)
        gradient_loss_inputs = np.dot(gradient_loss_input, gradient_input_inputs.T)

        self.gradients = {"weights": gradient_loss_weights, "biases": gradient_loss_bias}

        return gradient_loss_inputs

    def update_weights(self, gradients, learning_rate):
        self.weights -= learning_rate * gradients["weights"]
        self.biases -= learning_rate * gradients["biases"]

    def relu(self, x):
        return np.maximum(0, x)

    def d_relu(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        x = x - np.max(x, axis=1, keepdims=True)  # Stability
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

class Neural_Network:
    def __init__(self, inputs, len_layers, num_outputs):
        self.inputs = inputs
        self.len_layers = len_layers
        self.num_outputs = num_outputs
        self.layers = [
            Layer(784, len_layers),
            Layer(len_layers, len_layers - 50),
            Layer(len_layers - 50, 64),
            Layer(64, 16),
            Layer(16, num_outputs)
        ]
        # Added attributes from previous version
        self.trained = False
        self.num_epochs = 0
        self.accuracy = 0.0
        self.version = None

    def forward_pass(self, batch_images):
        batch_images = batch_images.reshape(batch_images.shape[0], -1)
        inputs = batch_images
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward_pass(self, labels):
        nn_predictions = self.layers[-1].activations
        gradient_loss_output = nn_predictions - labels
        for layer in reversed(self.layers):
            gradient_loss_output = layer.backward(gradient_loss_output)

    def cost(self, nn_predictions, labels):
        return -np.sum(labels * np.log(nn_predictions + 1e-7))

    def train(self, win, train_images, train_labels, num_epochs, batch_size, learning_rate, animation=True):
        if num_epochs is None:
            num_epochs = DEF_NUM_EPOCHS
        if batch_size is None:
            batch_size = DEF_BATCH_SIZE
        if learning_rate is None:
            learning_rate = DEF_LEARN_RATE

        num_samples = len(train_images)
        num_batches = num_samples // batch_size

        for epoch in range(num_epochs):
            correct_predictions = 0

            indices = np.random.permutation(num_samples)
            train_images = train_images[indices]
            train_labels = train_labels[indices]

            for i in range(num_batches):
                batch_images = train_images[i * batch_size:(i + 1) * batch_size]
                batch_labels = train_labels[i * batch_size:(i + 1) * batch_size]

                predictions = self.forward_pass(batch_images)

                loss = self.cost(predictions, batch_labels)

                if animation and i % 10 == 0:
                    draw_image(win, batch_images[0].reshape(28, 28))
                    pygame.display.update()

                self.backward_pass(batch_labels)
                for layer in self.layers:
                    layer.update_weights(layer.gradients, learning_rate)

                correct_predictions += np.sum(np.argmax(predictions, axis=1) == np.argmax(batch_labels, axis=1))
            accuracy = correct_predictions / num_samples

            # Update training visualization
            win.fill((0, 0, 0))
            draw_text(win, FONT, "TRAINING...", (20, 0), 100, (100, 100, 100))
            pygame.draw.line(win, (255, 255, 255), (0, 95), (WIDTH, 95), 9)
            epoch_str = f'Epoch {epoch+1}/{num_epochs}'
            accuracy_str = f'Accuracy = {accuracy:.4f}'
            loss_str = f'Loss = {loss:.4f}'
            draw_text(win, FONT, epoch_str, (0, 110), 30, (0, 0, 255))
            draw_text(win, FONT, accuracy_str, (0, 170), 30, (0, 0, 255))
            draw_text(win, FONT, loss_str, (0, 230), 30, (0, 0, 255))
            pygame.display.update()

        # Update network attributes
        self.trained = True
        self.num_epochs += num_epochs
        self.accuracy = accuracy

        # Final visualization update
        win.fill((0, 0, 0))
        if animation:
            draw_image(win, batch_images[0].reshape(28, 28))
        draw_text(win, FONT, "TRAINED!", (20, 0), 100, (100, 100, 100))
        pygame.draw.line(win, (255, 255, 255), (0, 95), (WIDTH, 95), 9)
        draw_text(win, FONT, epoch_str, (0, 110), 30, (255, 255, 255))
        draw_text(win, FONT, accuracy_str, (0, 170), 30, (255, 255, 255))
        draw_text(win, FONT, loss_str, (0, 230), 30, (255, 255, 255))
        pygame.display.update()
        pygame.time.delay(3000)

def draw_image(screen, image):
    width = image.shape[0]  
    pixel_size = 336 // width  

    start_x = WIDTH // 2 - 75
    start_y = HEIGHT // 2 - 75

    for row_index, row in enumerate(image):
        for col_index, pixel in enumerate(row):
            
            grayscale_value = int(pixel * 255)
            color = (grayscale_value, grayscale_value, grayscale_value)

            rect = pygame.Rect((col_index * pixel_size) + start_x, (row_index * pixel_size) + start_y,pixel_size, pixel_size )
            pygame.draw.rect(screen, color, rect)

    for i in range(width + 1): 
        y = start_y + i * pixel_size
        pygame.draw.line(screen, (200, 200, 200), (start_x, y), (start_x + width * pixel_size, y), 1)

    # Draw vertical grid lines
    for i in range(width + 1):  # Include the rightmost line
        x = start_x + i * pixel_size
        pygame.draw.line(screen, (200, 200, 200), (x, start_y), (x, start_y + width * pixel_size), 1)
        
#Helper Functions:
def draw_highlighted_rect(surface : pygame.surface.Surface, rect : pygame.rect.Rect, border_color : tuple, highlight_color : tuple, border_thickness : int, highlight_thickness : int):
    pygame.draw.rect(surface, border_color, rect, border_thickness)
    inner_rect = pygame.Rect(rect.left + border_thickness, rect.top + border_thickness,rect.width - 2 * border_thickness, rect.height - 2 * border_thickness)
    pygame.draw.rect(surface, highlight_color, inner_rect, highlight_thickness)

def draw_text(surface : pygame.surface.Surface, font : pygame.font.Font, text : str, pos : tuple, fontsize : int, color : tuple):
    font = pygame.font.Font(font, fontsize) # Font is reassigned as a pygame.Font obj to be blit.
    word = font.render(text, True, color)
    surface.blit(word, (pos[0], pos[1])) #word blit at right position in given font.

def reset_buttons(buttons : list):
    for button in buttons:
        button.clicked_ticks = 0
        button.clicked = False

def increment_button_ticks(buttons : list):
    for button in buttons:
        if button.clicked_ticks < FPS: #Uses FPS to ensure only close to a second is there for button delay.
            button.clicked_ticks += 1

def draw_menu(win, buttons, agent, clock):
    clock.tick(FPS)
    pygame.draw.line(win, (255, 255, 255), (0, 95), (WIDTH, 95), 9)
    draw_text(win, FONT, "MENU", (20, 0), 100, (100, 100, 100))
    if agent.trained:
        if agent.version != None:
            draw_text(win, FONT, f'version : {agent.version}', (320, 0), 30, (100, 100, 100))
        draw_text(win, FONT, f'accuracy : {agent.accuracy}', (320, 30), 30, (100, 100, 100))
        draw_text(win, FONT, f'epochs : {agent.num_epochs}', (320, 60), 30, (100, 100, 100))

    for button in buttons:
        button.draw(win)

    pygame.display.update()

def draw_train(win, buttons, start_button, save_button, image_button, clock):
    clock.tick(FPS)
    pygame.draw.line(win, (255, 255, 255), (0, 95), (WIDTH, 95), 9)
    draw_text(win, FONT, "TRAIN", (20, 0), 100, (100, 100, 100))
    draw_text(win, FONT, "Click to set value", (0, 100), 60, (100, 100, 150))

    for button in buttons:
        button.draw(win)

    start_button.draw(win)
    save_button.draw(win)
    image_button.draw(win)

    pygame.display.update()

# Function to load data
def get_data():
    with np.load('mnist.npz') as file:
        images, labels = file['x_train'], file['y_train']

    images = images.astype('float32') / 255
    labels = np.eye(10)[labels]

    print("Unique label values:", np.unique(labels))
    print("Label data type:", labels.dtype)

    return images, labels

def draw_load(win, agents, agent_buttons, clock):
    clock.tick(FPS)
    pygame.draw.line(win, (255, 255, 255), (0, 95), (WIDTH, 95), 9)
    draw_text(win, FONT, "LOAD", (20, 0), 100, (100, 100, 100))

    if len(agents) >= 1:
        for agent in agents:
            agent_buttons[agent - 1].draw(win)
            x, y = agent_buttons[agent - 1].rect.x, agent_buttons[agent - 1].rect.y
            draw_text(win, FONT, f'{agents[agent].accuracy}', (x + 235, y), 40, (255, 255, 255))
            draw_text(win, FONT, f'{agents[agent].num_epochs}', (x + 420, y), 40, (255, 255, 255))

        draw_text(win, FONT, "Version", (40, 100), 40, (255, 255, 255))
        draw_text(win, FONT, "Accuracy", (225, 100), 40, (255, 255, 255))
        draw_text(win, FONT, "Epochs", (410, 100), 40, (255, 255, 255))
        pygame.draw.line(win, (255, 255, 255), (0, 130), (WIDTH, 130), 4)
        pygame.draw.line(win, (255, 255, 255), (385, 95), (385, HEIGHT), 4)
        pygame.draw.line(win, (255, 255, 255), (220, 95), (220, HEIGHT), 4)

    else:
        draw_text(win, FONT, "No agents saved", (15, HEIGHT//2 - 75), 75, (255, 0, 0))

    pygame.display.update()

def draw_window(win, ROWS, COLS, pixels, SQUARE_SIZE, state, menu_buttons, train_buttons, 
                start_button, save_button, agent_buttons, random_button, image_button, clock, agents, prediction, agent):
    if state == "Drawing":
        clock.tick(FPS)
        win.fill((100, 100, 100))
        for j in range(ROWS):
            for i in range(COLS):
                grayscale_value = int(pixels[i, j] * 255)
                color = (grayscale_value, grayscale_value, grayscale_value)
                pygame.draw.rect(win, color, (j * SQUARE_SIZE, i * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

        percentages = prediction[0]

        for percent in range(len(percentages)):
            draw_text(win, FONT, f'{percent} : {percentages[percent] * 100}%', (290, percent*53), 50, (0, 0, 0))

        random_button.draw(win)
        draw_text(win, FONT, f'PREDICTED - {np.argmax(percentages)}', (0, 290), 30, (0, 139, 139))
        pygame.draw.line(win, (255, 255, 255), (285, 0), (285, HEIGHT), 5)

        pygame.display.update()

    elif state == "Controls":
        clock.tick(FPS)
        draw_text(win, FONT, "Controls", (20, 0), 100, (100, 100, 100))
        pygame.draw.line(win, (255, 255, 255), (0, 95), (WIDTH, 95), 9)
        draw_text(win, FONT, "LEFT CLICK : Click/Draw", (10, 100), 40, (255, 255, 255))
        draw_text(win, FONT, "RIGHT CLICK : Rub / Delete Save", (10, 150), 40, (255, 255, 255))
        draw_text(win, FONT, "R : Reset Board", (10, 200), 40, (255, 255, 255))
        draw_text(win, FONT, "ESCAPE : Exit any state", (10, 250), 40, (255, 255, 255))
        draw_text(win, FONT, "SPACE : Get a prediction", (10, 300), 40, (255, 255, 255))
        draw_text(win, FONT, "LOAD : to load, click version", (10, 350), 40, (255, 255, 255))
        draw_text(win, FONT, "TOGGLE GREEN : ON", (10, 400), 40, (255, 255, 255))
        draw_text(win, FONT, "TOGGLE RED : OFF", (10, 450), 40, (255, 255, 255))

        pygame.display.update()

    elif state == "Train":
        draw_train(win, train_buttons, start_button, save_button, image_button, clock)

    elif state == "Menu":
        draw_menu(win, menu_buttons, agent, clock)

    elif state == "Load":
        draw_load(win, agents, agent_buttons, clock)

def save_agent(agent, directory="NNagents"):
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    existing_files = [f for f in os.listdir(directory) if f.startswith("NNdata") and f.endswith(".pickle")]
    existing_versions = [int(f.replace("NNdata", "").replace(".pickle", "")) for f in existing_files if f[6:-7].isdigit()]
    
    agent.version = max(existing_versions, default=0) + 1
    filename = f"NNdata{agent.version}.pickle"
    filepath = os.path.join(directory, filename)
    
    data = {
        "inputs": agent.inputs,
        "len_layers": agent.len_layers,
        "num_outputs": agent.num_outputs,
        "trained": agent.trained,
        "num_epochs": agent.num_epochs,
        "version": agent.version,
        "accuracy": agent.accuracy,
        "layers": [{
            "weights": layer.weights,
            "biases": layer.biases,
            "num_inputs": layer.num_inputs,
            "num_outputs": layer.num_outputs
        } for layer in agent.layers]
    }
    
    with open(filepath, "wb") as f:
        pickle.dump(data, f)

def load_agents(directory="NNagents"):
    agents = {}

    if not os.path.exists(directory):
        return agents

    # Iterate through all files matching the naming convention
    for file in [f for f in os.listdir(directory) if f.startswith("NNdata") and f.endswith(".pickle")]:
        filepath = os.path.join(directory, file)
        try:
            with open(filepath, "rb") as f:
                data = pickle.load(f)
            
            # Create and configure a new agent from the loaded data
            agent = Neural_Network(
                inputs=data.get("inputs", []),  # Default to an empty list if not provided
                len_layers=data["len_layers"],
                num_outputs=data["num_outputs"]
            )
            for i, layer_data in enumerate(data["layers"]):
                agent.layers[i].weights = layer_data["weights"]
                agent.layers[i].biases = layer_data["biases"]

            agent.trained = data["trained"]
            agent.num_epochs = data["num_epochs"]
            agent.version = data.get("version", len(agents) + 1)  # Assign a version if not present
            agent.accuracy = data["accuracy"]

            agents[agent.version] = agent

        except Exception as e:
            print(f"Failed to load agent from {file}: {e}")
            continue

    # Return a dictionary of agents, sorted by version number
    return dict(sorted(agents.items()))

def main():

    clock = pygame.time.Clock()

    pixels = np.zeros((ROWS, COLS))
    win = pygame.display.set_mode((WIDTH, HEIGHT))

    train_button = Button(0, 101, 200, 100, "Train", 75, (0, 139, 139), (0, 139, 139))
    controls_button = Button(0, 201, 340, 100, "Controls", 75, (255, 255, 255), (255, 255, 255))
    load_button = Button(0, 301, 190, 100, "Load", 75, (183, 65, 14), (183, 65, 14))
    draw_button = Button(0, 401, 190, 100, "Draw", 75, (10, 50, 10), (10, 50, 10))

    epoch_button = Button(0, 175, WIDTH, 75, "Num Epochs: ", 60, (0, 139, 139), (0, 139, 139), thickness=3)
    batch_button = Button(0, 255, WIDTH, 75, "Batch Size: ", 60, (0, 139, 139), (0, 139, 139), thickness=3)
    learning_button = Button(0, 335, WIDTH, 75, "Learn rate: ", 60, (0, 139, 139), (0, 139, 139), thickness=3)
    start_button = Button(10, 415, 210, 80, "START", 75, (100, 100, 100), (100, 100, 100))
    save_button = Button(250, 415, 210, 80, "SAVE", 75, (100, 100, 100), (100, 100, 100))

    image_button = Button(WIDTH - 310, HEIGHT - 40, 310, 40, "ANIMATION TOGGLE", 40, (100, 100, 100), (100, 100, 100), thickness=3)

    agent_buttons = []

    random_button = Button(0, 460, 280, 80, "RANDOM", 75, (0, 0, 0), (0, 0, 0))

    menu_buttons = [train_button, controls_button, load_button, draw_button]
    
    train_buttons = [epoch_button, batch_button, learning_button]
    state = "Menu"

    selected_button = None
    editing_string = ""
    selected_save = None

    num_epochs = None
    batch_size = None
    learning_rate = None
    agent = Neural_Network([], 128, 10)
    images, labels = get_data()
    prediction = [np.zeros(10)]

    animation = True

    agents = load_agents()

    current_keys = sorted(agents.keys())

    if current_keys != list(range(1, len(agents) + 1)):
        
        agents = {i + 1: v for i, (_, v) in enumerate(sorted(agents.items()))}
        
        agent_buttons = []
        for i, _ in agents.items():
            string = f'Network : {i}'
            agent_buttons.append(Button(0, 150 + 55 * (i - 1), 340, 55, string, 40, thickness=3, textcolor=(100, 100, 100)))
    else:
        for a in agents:
            string = f'Network : {a}'
            agent_buttons.append(Button(0, 150 + 55*(a-1), 340, 55, string, 40, thickness=3, textcolor=(100, 100, 100)))

    # Main loop
    run = True
    drawing = False  
    
    while run:
        win.fill((0, 0, 0))
        draw_window(win, ROWS, COLS, pixels, SQUARE_SIZE, state, menu_buttons, train_buttons, start_button, save_button, agent_buttons, random_button, image_button, clock, agents, prediction, agent)
        pygame.display.set_caption(f"CNN - {str(int(clock.get_fps()))}")
        increment_button_ticks(menu_buttons)
        increment_button_ticks(train_buttons)
        increment_button_ticks([start_button, save_button, random_button, image_button])
        increment_button_ticks(agent_buttons)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            try:
                if event.type == pygame.MOUSEBUTTONDOWN:

                    if state == "Drawing":
                        if event.button == 1: 
                            drawing = True

                        elif event.button == 3: 
                            x, y = pygame.mouse.get_pos()
                            i, j = y // SQUARE_SIZE, x // SQUARE_SIZE
                            pixels[i, j] = 0  

                    elif state == "Load":
                        if event.button == 3:
                            for button in agent_buttons:
                                if button.rect.collidepoint(event.pos):
                                    
                                    save_file = "NNagents\\" + f"NNdata{agents[agent_buttons.index(button) + 1].version}.pickle"

                                    os.remove(save_file)

                                    agent_buttons.pop(agent_buttons.index(button))
                                    agents.pop(int(button.text[-1]))

                            if current_keys != list(range(1, len(agents) + 1)):
        
                                agents = {i + 1: v for i, (_, v) in enumerate(sorted(agents.items()))}
                                
                                agent_buttons = []
                                for i, _ in agents.items():
                                    string = f'Network : {i}'
                                    agent_buttons.append(Button(0, 150 + 55 * (i - 1), 340, 55, string, 40, thickness=3, textcolor=(100, 100, 100)))

                elif event.type == pygame.MOUSEBUTTONUP and state == "Drawing":
                    if event.button == 1:  
                        drawing = False

                elif event.type == pygame.MOUSEMOTION and state == "Drawing":
                    if drawing:
                        x, y = pygame.mouse.get_pos()
                        i, j = y // SQUARE_SIZE, x // SQUARE_SIZE
                        pixels[i, j] = 1 

                elif event.type == pygame.KEYDOWN:
                    if state == "Drawing":
                        if event.key == pygame.K_SPACE:
                            user_image = pixels.reshape(1, 784).astype('float32')

                            prediction = agent.forward_pass(user_image)
                            print(f'Predicted number: {np.argmax(prediction)}')
                            percentages = prediction[0]
                            for percent in range(len(percentages)):
                                print(f"Number = {percent}, Probability = {percentages[percent] * 100:.2f}%")
                            
                        elif event.key == pygame.K_r:  
                            pixels.fill(0)

                        elif event.key == pygame.K_ESCAPE:
                            state = "Menu"

                    elif state == "Train":
                        if selected_button != None:
                            if event.key == pygame.K_ESCAPE:
                                train_buttons[selected_button].textcolor = (0, 139, 139)
                                selected_button = None
                                editing_string = ""

                            else:
                                if event.key == pygame.K_BACKSPACE:
                                    if len(train_buttons[selected_button].text) - 1 >= train_buttons[selected_button].text_len:
                                        train_buttons[selected_button].text = train_buttons[selected_button].text[:-1]
                                else:
                                    editing_string += event.unicode

                                try:
                                    user_val = float((train_buttons[selected_button].text + editing_string)[train_buttons[selected_button].text_len:])
                                    user_string = train_buttons[selected_button].text[train_buttons[selected_button].text_len:]
                                    if selected_button == 0:  # Epochs
                                        if len(user_string) >= 7:
                                            raise ValueError
                                        num_epochs = int(user_val)
                                    elif selected_button == 1:  # Batch Size
                                        if len(user_string) >= 8:
                                            raise ValueError
                                        batch_size = int(user_val)
                                    elif selected_button == 2:  # Learning Rate
                                        if len(user_string) >= 7:
                                            raise ValueError
                                        learning_rate = user_val

                                    train_buttons[selected_button].text += editing_string

                                except ValueError:
                                    pass
                                editing_string = ""

                                if event.key == pygame.K_ESCAPE:
                                    state = "Menu"
                        else:
                            if event.key == pygame.K_ESCAPE:
                                state = "Menu"

                    elif state == "Controls" or state == "Load":
                        if event.key == pygame.K_ESCAPE:
                            state = "Menu"

            except IndexError:
                pass

        if state == "Train":
            start_button.get_clicked()
            image_button.get_clicked()
            if agent.trained and len(agents) < 7:
                save_button.get_clicked()

            if start_button.clicked:
                reset_buttons([start_button])

                win.fill((0, 0, 0))

                draw_text(win, FONT, "TRAINING...", (20, 0), 100, (100, 100, 100))
                pygame.draw.line(win, (255, 255, 255), (0, 95), (WIDTH, 95), 9)
                epoch_str, accuracy_str, loss_str = f'Epoch 0/{num_epochs}', f'Accuracy = 0.0', f'Loss = 0.0'
                draw_text(win, FONT, epoch_str, (0, 110), 30, (0, 0, 255))
                draw_text(win, FONT, accuracy_str, (0, 170), 30, (0, 0, 255))
                draw_text(win, FONT, loss_str, (0, 230), 30, (0, 0, 255))
                pygame.display.update()
                agent.train(win, images, labels, num_epochs, batch_size, learning_rate, animation)

            if save_button.clicked:
                reset_buttons([save_button])
                save_agent(agent)
                agents = load_agents()

                if current_keys != list(range(1, len(agents) + 1)):
        
                    agents = {i + 1: v for i, (_, v) in enumerate(sorted(agents.items()))}
                    
                    agent_buttons = []
                    for i, _ in agents.items():
                        string = f'Network : {i}'
                        agent_buttons.append(Button(0, 150 + 55 * (i - 1), 340, 55, string, 40, thickness=3, textcolor=(100, 100, 100)))
                else:
                    agent_buttons = [Button(0, 150 + 55*i, 340, 55, f'Network : {i+1}', 40, thickness=3, textcolor=(100, 100, 100))for i in range(len(agents))]
                selected_save = len(agent_buttons) - 1

            if selected_button == None:
                for button in train_buttons:
                    button.get_clicked()
                    if button.clicked:
                        reset_buttons(train_buttons)
                        button.textcolor = SELECT_COLOR 
                        selected_button = train_buttons.index(button)

            if image_button.clicked:
                reset_buttons([image_button])
                if animation:
                    image_button.textcolor = (255, 0, 0)
                    animation = False
                else:
                    image_button.textcolor = (0, 255, 0)
                    animation = True

        elif state == "Load":
            for button in agent_buttons:
                button.get_clicked()

                if button.clicked:
                    reset_buttons(agent_buttons)
                    selected_save = agent_buttons.index(button)

                    agent = agents[agent_buttons.index(button) + 1]

                if agent_buttons.index(button) == selected_save:
                    button.textcolor = (255, 0, 0)

                else:
                    button.textcolor = (100, 100, 100)

        elif state == "Drawing":
            random_button.get_clicked()
            if random_button.clicked:
                reset_buttons([random_button])
                image = images[np.random.randint(0, 10000)]
                pixels = image

        #state changes:
        elif state == "Menu":
            for button in menu_buttons:
                button.get_clicked()
            
            if controls_button.clicked:
                reset_buttons(menu_buttons)
                state = "Controls"

            elif train_button.clicked:
                reset_buttons(menu_buttons)
                state = "Train"

            elif load_button.clicked:
                reset_buttons(menu_buttons)
                state = "Load"

            elif draw_button.clicked:
                reset_buttons(menu_buttons)
                state = "Drawing"

    pygame.quit()

if __name__ == "__main__":
    main()