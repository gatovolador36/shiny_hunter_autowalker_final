import mss
import numpy as np
import pygetwindow as gw
import pyautogui
import asyncio
import cv2
import os
from playsound import playsound
import time
from PIL import Image

def is_in_battle(pixel_data: np.ndarray) -> bool:
    """Checks if a battle is active by scanning the entire top screen for white pixels."""
    white_threshold = 255  # Exact white color
    min_white_pixels = 230  # Minimum number of white pixels to consider it as in battle

    # Only consider the center area for battle detection to avoid false positives
    center_area = pixel_data[50:200, 50:200, :]

    white_pixels = np.all(center_area == white_threshold, axis=2)
    num_white_pixels = np.count_nonzero(white_pixels)

    return num_white_pixels >= min_white_pixels

def ensure_same_type_and_depth(image1, image2):
    """Ensure that both images are of the same type and depth."""
    if image1.dtype != image2.dtype:
        image2 = image2.astype(image1.dtype)
    return image1, image2

def ensure_four_channels(image):
    """Ensure the image has four channels (including alpha channel)."""
    if image.shape[2] == 3:  # If the image has three channels, add an alpha channel
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    return image

def convert_to_cv8u(image):
    """Convert the image to CV_8U type."""
    if image.dtype != np.uint8:
        image = cv2.convertScaleAbs(image)
    return image

def preprocess_image(image):
    """Preprocess the image for template matching."""
    image = ensure_four_channels(image)
    image = convert_to_cv8u(image)
    return image

def create_shiny_mask(shiny_sprites):
    """Create a mask that isolates the shiny sprites."""
    masks = []
    for sprite in shiny_sprites:
        gray = cv2.cvtColor(sprite, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        masks.append(mask)
    return masks

def isolate_sprite(roi: np.ndarray, shiny_sprites: list) -> np.ndarray:
    """Isolate the sprite in the region of interest (ROI) by removing the background."""
    masks = create_shiny_mask(shiny_sprites)
    combined_mask = np.zeros((roi.shape[0], roi.shape[1]), dtype=np.uint8)
    for mask in masks:
        mask_resized = cv2.resize(mask, (roi.shape[1], roi.shape[0]))  # Ensure size match
        combined_mask = cv2.bitwise_or(combined_mask, mask_resized)
    combined_mask_inv = cv2.bitwise_not(combined_mask)
    sprite = cv2.bitwise_and(roi, roi, mask=combined_mask_inv)
    return sprite

def extract_frames_from_gif(gif_path):
    """Extract frames from a GIF file."""
    gif = Image.open(gif_path)
    frames = []
    try:
        while True:
            frame = gif.copy()
            frame = frame.convert("RGBA")
            frame = np.array(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGRA)
            frames.append(frame)
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass
    return frames

def load_shiny_sprites(sprite_folders):
    """Load shiny sprites and extract frames from GIFs."""
    shiny_sprites = []
    for folder in sprite_folders:
        for sprite_file in os.listdir(folder):
            sprite_path = os.path.join(folder, sprite_file)
            if sprite_file.endswith('.gif'):
                frames = extract_frames_from_gif(sprite_path)
                shiny_sprites.extend(frames)
            else:
                shiny_sprite = cv2.imread(sprite_path, cv2.IMREAD_UNCHANGED)  # Load with alpha channel
                if shiny_sprite is not None:
                    shiny_sprites.append(shiny_sprite)
    return shiny_sprites

def is_shiny(pixel_data: np.ndarray, shiny_sprites: list) -> bool:
    """Checks if the encountered Pokémon is shiny by comparing with shiny sprites."""
    capture_file = 'current_encounter.png'
    
    # Save the encounter screenshot
    cv2.imwrite(capture_file, cv2.cvtColor(pixel_data, cv2.COLOR_RGB2BGR))
    
    # Preprocess the captured frame
    preprocessed_frame = preprocess_image(pixel_data)
    cv2.imwrite('frame_preprocessed.png', preprocessed_frame)  # Save preprocessed frame for debugging
    
    for shiny_sprite in shiny_sprites:
        # Preprocess the shiny sprite
        shiny_sprite = preprocess_image(shiny_sprite)
        
        # Ensure both images are of the same type and depth
        preprocessed_frame, shiny_sprite = ensure_same_type_and_depth(preprocessed_frame, shiny_sprite)
        
        # Compare images using color
        result = cv2.matchTemplate(preprocessed_frame, shiny_sprite, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        
        print(f"Comparing with shiny sprite: max_val = {max_val}")
        
        if max_val > 0.7:  # Threshold for match, slightly lowered
            print("Shiny match found!")
            return True
    return False

async def run_from_battle(window) -> None:
    """Simulates key presses to run from battle."""
    print("Running from battle...")
    window.activate()
    await asyncio.sleep(0.5)  # Give some time for the window to activate
    keys = ['w', 'a', 'd', 'enter']
    for key in keys:
        print(f"Pressing {key}")
        pyautogui.keyDown(key)
        await asyncio.sleep(0.2)  # Hold the key down for a brief moment
        pyautogui.keyUp(key)
        await asyncio.sleep(0.7)  # Interval between key presses
    print("Key presses sent to run from battle.")

async def handle_shiny_encounter(window) -> None:
    """Handles actions to take when a shiny Pokémon is encountered."""
    print("Shiny Pokémon detected!")
    window.activate()
    await asyncio.sleep(0.5)  # Give some time for the window to activate
    # Play sound
    sound_file = 'shiny_sound.mp3'
    if os.path.exists(sound_file):
        try:
            print(f"Playing sound: {sound_file}")
            playsound(sound_file)
            print("Shiny sound played. Program will cease operation.")
        except Exception as e:
            print(f"Error playing sound: {e}")
    else:
        print(f"Error: Sound file '{sound_file}' not found.")
    exit(0)

async def autowalk() -> None:
    """Main function for the autowalker and shiny hunter."""
    print("Autowalker and Shiny Hunter starting...")

    # --- Configuration ---
    emulator_title: str = "DeSmuME 0.9.13 x64 SSE2"
    walk_duration: float = 0.5  # Duration to hold each key down
    rest_duration: float = 1.0  # Duration to rest between key presses
    capture_width: int = 250     # Capture width for the entire top screen (battle detection)
    capture_height: int = 250    # Capture height for the entire top screen (battle detection)
    capture_delay: float = 0.08
    battle_detect_delay: float = 10.0  # Delay before running from battle
    post_battle_delay: float = 7.0     # Delay after running from battle
    key_press_interval: float = 0.7    # Interval between key presses
    frame_comparison_limit: int = 5    # Limit the number of frames to compare for shiny detection
    battle_timeout: float = 60.0       # Timeout for battle detection to prevent infinite looping

    # Load shiny sprites
    sprite_folders = ['shiny_sprites', 'shiny_sprites_female']
    shiny_sprites = load_shiny_sprites(sprite_folders)

    # --- Initialization ---
    try:
        window = gw.getWindowsWithTitle(emulator_title)[0]
        with mss.mss() as sct:
            capture_region_battle = {
                "top": window.top,
                "left": window.left,
                "width": capture_width,
                "height": capture_height,
                "mon": 1,  # Assuming primary monitor
            }

            # --- Main Loop ---
            print("Autowalker running...")
            while True:
                try:
                    window.activate()
                except IndexError:
                    print(f"Error: Could not find window with title '{emulator_title}'.")
                    return

                # Delay to ensure the Pokémon is fully visible
                await asyncio.sleep(0.3)
                
                # Capture multiple frames during the battle
                frames = []
                for frame_idx in range(20):  # Capture 20 frames
                    sct_img = sct.grab(capture_region_battle)
                    img_array = np.array(sct_img)
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGRA2BGR)  # Convert BGRA to BGR to ensure correct color order
                    frames.append(img_array)
                    cv2.imwrite(f'frame_{frame_idx}.png', img_array)  # Save each captured frame for debugging
                    await asyncio.sleep(0.05)  # Small delay between captures

                # Check for battle
                battle_detected = is_in_battle(frames[0])
                print(f"Battle detected: {battle_detected}")
                if battle_detected:
                    print("Battle detected. Waiting before running...")
                    await asyncio.sleep(battle_detect_delay)  # Delay before running from battle

                    # Check each captured frame for shiny with a limit
                    shiny_detected = False
                    start_time = time.time()
                    for frame_idx, frame in enumerate(frames[:frame_comparison_limit]):
                        print(f"Checking frame {frame_idx + 1} for shiny...")
                        shiny_detected = is_shiny(frame, shiny_sprites)
                        if shiny_detected:
                            break
                        if time.time() - start_time > battle_timeout:
                            print("Battle detection timeout reached. Running from battle...")
                            break

                    if shiny_detected:
                        await handle_shiny_encounter(window)
                        break  # Exit the loop after handling the shiny encounter
                    else:
                        print("No shiny detected. Running from battle...")
                        await run_from_battle(window)
                        await asyncio.sleep(post_battle_delay)  # Wait for the battle to end and resume walking

                # Simulate walking in a pattern (up, right, down, left)
                pyautogui.keyDown('w')  # Press 'w' key to move up
                await asyncio.sleep(walk_duration)
                pyautogui.keyUp('w')
                await asyncio.sleep(key_press_interval)

                pyautogui.keyDown('d')  # Press 'd' key to move right
                await asyncio.sleep(walk_duration)
                pyautogui.keyUp('d')
                await asyncio.sleep(key_press_interval)

                pyautogui.keyDown('s')  # Press 's' key to move down
                await asyncio.sleep(walk_duration)
                pyautogui.keyUp('s')
                await asyncio.sleep(key_press_interval)

                pyautogui.keyDown('a')  # Press 'a' key to move left
                await asyncio.sleep(walk_duration)
                pyautogui.keyUp('a')
                await asyncio.sleep(key_press_interval)

                await asyncio.sleep(capture_delay)

    except KeyboardInterrupt:
        print("Autowalker stopped.")
    except Exception as e:
        print(f"An error occurred in Autowalker: {e}")

if __name__ == "__main__":
    asyncio.run(autowalk())