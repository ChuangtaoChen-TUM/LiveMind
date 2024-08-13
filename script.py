import matplotlib.pyplot as plt
import matplotlib.patches as patches

def render_text_background(texts: list[str], colors: list[str]|None=None, color_map=None, max_width=0.5, output=None):
    if colors is not None:
        if len(texts) != len(colors):
            raise ValueError("The number of texts and colors must be equal.")
    if color_map is None:
        # Default color map if none is provided
        color_map = ['blue', 'yellow', 'green', 'red', 'purple', 'orange']

    # Define the font size and padding
    fontsize = 9
    line_height = fontsize * 0.004  # Approximate height of a line
    padding = 0  # Padding between text segments

    def render_text(ax, texts, color_map, max_width):
        # Initialize the starting position
        x_position = 0.1
        y_position = 0.5

        for i, text in enumerate(texts):
            if colors is not None:
                color = colors[i]
            else:
                color = color_map[i % len(color_map)]
            
            
            if x_position > max_width:
                text = text.lstrip()
    
            # Create the text object to measure its size
            text_artist = ax.text(
                x_position,
                y_position,
                text,
                fontsize=fontsize,
                fontname='monospace',
                ha='left', va='center'
            )
            fig.canvas.draw()  # Draw the canvas to update the renderer

            # Get the bounding box of the text
            bbox = text_artist.get_window_extent(renderer=fig.canvas.get_renderer())
            bbox = bbox.transformed(ax.transData.inverted())

            # Check if the text exceeds the max_width
            if x_position > max_width:
                # Move to the next line
                x_position = 0.1
                y_position -= line_height

                # Update the text position
                text_artist.set_position((x_position, y_position))
                fig.canvas.draw()  # Draw the canvas to update the renderer

                # Get the updated bounding box of the text
                bbox = text_artist.get_window_extent(renderer=fig.canvas.get_renderer())
                bbox = bbox.transformed(ax.transData.inverted())

            # Create a rectangle patch with the same size as the text bounding box
            rect = patches.FancyBboxPatch((bbox.x0, bbox.y0), bbox.width, bbox.height,
                                          boxstyle="round,pad=0", facecolor=color, edgecolor='none', zorder=-1)

            # Add the rectangle patch to the axes
            ax.add_patch(rect)

            # Redraw the text on top of the rectangle
            text_artist.set_zorder(1)

            # Update the x_position for the next text segment
            x_position += bbox.width + padding

    # Create a new figure and axis for the PDF
    fig, ax = plt.subplots()
    ax.axis('off')
    render_text(ax, texts, color_map, max_width)
    if output is None:
        plt.show()
    else:
        if output.endswith('.svg'):
            plt.savefig(output, format='svg')
        elif not output.endswith('.pdf'):
            plt.savefig(output + '.pdf', format='pdf')
        else:
            plt.savefig(output, format='pdf')
    plt.close()

from live_mind.text.segmenter import get_segmenter

color_map = ['lightblue', 'lightyellow', 'lightpink', 'lightgreen', 'lightcoral', 'lavender',]
# Example usage
text = "Construct a complete truth table for the following argument. Then, using the truth table, determine whether the argument is valid or invalid. If the argument is invalid, choose an option which presents a counterexample."

sents = get_segmenter("sent")(text)
texts = []
colors = []

for i, sent in enumerate(sents):
    words = get_segmenter("word")(sent)
    texts.extend(words)
    colors.extend([color_map[i % len(color_map)] for _ in words])

render_text_background(texts, colors=colors, max_width=0.8, output='render_sent.svg')


clauses = get_segmenter("clause")(text)
texts = []
colors = []

for i, clause in enumerate(clauses):
    words = get_segmenter("word")(clause)
    texts.extend(words)
    colors.extend([color_map[i % len(color_map)] for _ in words])

render_text_background(texts, colors=colors, max_width=0.8, output='render_clause.svg')


words = get_segmenter("word")(text)
texts = []
colors = []

for i in range(len(words)):
    texts.append(words[i])
    colors.append(color_map[i % len(color_map)])

render_text_background(texts, colors=colors, max_width=0.8, output='render_word.svg')

chars = get_segmenter("char")(text)
texts = []
colors = []

for i in range(len(chars)):
    texts.append(chars[i])
    colors.append(color_map[i % len(color_map)])

render_text_background(texts, colors=colors, max_width=0.8, output='render_char.svg')