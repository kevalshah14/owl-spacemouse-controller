#!/usr/bin/env python3
"""
Simple pygame script to control a ball with SpaceMouse.
"""

import pygame
import sys

try:
    import pyspacemouse
except ImportError:
    print("Error: pyspacemouse not installed. Run: uv pip install pyspacemouse")
    sys.exit(1)

# Settings
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
BALL_RADIUS = 20
SENSITIVITY = 8.0  # How fast the ball moves
DEADZONE = 0.08    # Ignore small inputs

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 80, 80)
GREEN = (80, 255, 80)
BLUE = (80, 80, 255)
BALL_COLOR = (255, 150, 50)  # Orange


def main():
    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("SpaceMouse Ball Control")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)
    small_font = pygame.font.Font(None, 24)
    
    # Initialize SpaceMouse
    spacemouse = pyspacemouse.open()
    if not spacemouse:
        print("SpaceMouse not found!")
        pygame.quit()
        sys.exit(1)
    
    print(f"SpaceMouse connected: {spacemouse.product_name}")
    
    # Ball position (center of screen)
    ball_x = WINDOW_WIDTH / 2
    ball_y = WINDOW_HEIGHT / 2
    ball_z = 0  # Z affects ball size
    
    # Trail effect
    trail = []
    max_trail = 50
    
    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    # Reset position
                    ball_x = WINDOW_WIDTH / 2
                    ball_y = WINDOW_HEIGHT / 2
                    ball_z = 0
                    trail.clear()
        
        # Read SpaceMouse
        state = spacemouse.read()
        
        # Apply deadzone (translation only)
        raw_x = state.x if abs(state.x) > DEADZONE else 0
        raw_y = state.y if abs(state.y) > DEADZONE else 0
        raw_z = state.z if abs(state.z) > DEADZONE else 0
        
        # Axis lock - only move in the strongest axis
        abs_x, abs_y, abs_z = abs(raw_x), abs(raw_y), abs(raw_z)
        max_axis = max(abs_x, abs_y, abs_z)
        
        if max_axis > 0:
            # Only allow movement on the dominant axis
            sm_x = raw_x if abs_x == max_axis else 0
            sm_y = raw_y if abs_y == max_axis else 0
            sm_z = raw_z if abs_z == max_axis else 0
        else:
            sm_x = sm_y = sm_z = 0
        
        # Update ball position
        ball_x += sm_x * SENSITIVITY
        ball_y -= sm_y * SENSITIVITY  # Inverted for screen coords
        ball_z += sm_z * SENSITIVITY * 0.15  # Reduced Z sensitivity
        
        # Clamp to screen bounds
        ball_x = max(BALL_RADIUS, min(WINDOW_WIDTH - BALL_RADIUS, ball_x))
        ball_y = max(BALL_RADIUS, min(WINDOW_HEIGHT - BALL_RADIUS, ball_y))
        ball_z = max(-50, min(50, ball_z))
        
        # Add to trail
        if sm_x != 0 or sm_y != 0:
            trail.append((int(ball_x), int(ball_y)))
            if len(trail) > max_trail:
                trail.pop(0)
        
        # Calculate ball size based on Z (closer = bigger)
        current_radius = int(BALL_RADIUS + ball_z * 0.3)
        current_radius = max(5, current_radius)
        
        # Clear screen
        screen.fill(BLACK)
        
        # Draw grid
        for x in range(0, WINDOW_WIDTH, 50):
            pygame.draw.line(screen, (30, 30, 30), (x, 0), (x, WINDOW_HEIGHT))
        for y in range(0, WINDOW_HEIGHT, 50):
            pygame.draw.line(screen, (30, 30, 30), (0, y), (WINDOW_WIDTH, y))
        
        # Draw axes at center
        center_x, center_y = WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2
        pygame.draw.line(screen, RED, (center_x - 50, center_y), (center_x + 50, center_y), 2)
        pygame.draw.line(screen, GREEN, (center_x, center_y - 50), (center_x, center_y + 50), 2)
        
        # Draw trail
        for i, pos in enumerate(trail):
            alpha = int(255 * (i / max_trail))
            trail_color = (alpha // 2, alpha // 3, alpha // 4)
            pygame.draw.circle(screen, trail_color, pos, 3)
        
        # Draw ball shadow
        shadow_offset = int(ball_z * 0.2)
        pygame.draw.circle(screen, (40, 40, 40), 
                          (int(ball_x) + shadow_offset, int(ball_y) + shadow_offset), 
                          current_radius)
        
        # Draw ball
        pygame.draw.circle(screen, BALL_COLOR, (int(ball_x), int(ball_y)), current_radius)
        pygame.draw.circle(screen, WHITE, (int(ball_x), int(ball_y)), current_radius, 2)
        
        # Draw highlight
        highlight_x = int(ball_x - current_radius * 0.3)
        highlight_y = int(ball_y - current_radius * 0.3)
        pygame.draw.circle(screen, (255, 200, 150), (highlight_x, highlight_y), current_radius // 4)
        
        # Draw info text
        title = font.render("SpaceMouse Ball Control", True, WHITE)
        screen.blit(title, (10, 10))
        
        pos_text = small_font.render(f"Position: X={ball_x:.0f}  Y={ball_y:.0f}  Z={ball_z:.0f}", True, WHITE)
        screen.blit(pos_text, (10, 50))
        
        # Show translation inputs with active axis highlighted
        active_axis = "X" if sm_x != 0 else ("Y" if sm_y != 0 else ("Z" if sm_z != 0 else "-"))
        trans_text = small_font.render(f"Input:  X={sm_x:+.2f}  Y={sm_y:+.2f}  Z={sm_z:+.2f}  [Axis: {active_axis}]", True, (150, 200, 150))
        screen.blit(trans_text, (10, 75))
        
        controls = small_font.render("R = Reset | ESC = Quit", True, (100, 100, 100))
        screen.blit(controls, (10, WINDOW_HEIGHT - 30))
        
        # Axis labels
        x_label = small_font.render("X", True, RED)
        y_label = small_font.render("Y", True, GREEN)
        screen.blit(x_label, (center_x + 55, center_y - 10))
        screen.blit(y_label, (center_x - 5, center_y - 70))
        
        # Update display
        pygame.display.flip()
        clock.tick(60)
    
    # Cleanup
    spacemouse.close()
    pygame.quit()
    print("Done!")


if __name__ == "__main__":
    main()
