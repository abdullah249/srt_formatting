import re
from datetime import datetime, timedelta
import os
import hashlib
import shutil
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class RecursiveSRTFormatter:
    def __init__(self):
        self.min_duration_short = 0.3  # Minimum duration for short subtitles (1-2 words)
        self.min_duration = 0.8  # Minimum duration for longer subtitles
        self.max_duration = 6.0
        self.max_chars_per_line = 45
        self.min_words_per_block = 2
        self.max_lines_per_block = 2
        self.max_iterations = 3
        self.seen_blocks = set()
        self.sentence_endings = ['.', '!', '?', '...']
        self.last_end_time = None
        self.min_gap = 0.1
        self.chunk_size = 1500
        self.test_mode = False
        self.test_range = (1520, 1540)
        self.cps = 15.0  # Characters per second for duration calculation

    def balance_lines(self, block_text):
        """Ensure no single word is alone on the second line of a block"""
        lines = block_text.split('\n')
        if len(lines) != 2:
            return block_text
        line1_words = lines[0].split()
        line2_words = lines[1].split()
        if len(line2_words) == 1:
            all_words = line1_words + line2_words
            midpoint = len(all_words) // 2
            new_line1 = ' '.join(all_words[:midpoint])
            new_line2 = ' '.join(all_words[midpoint:])
            if len(new_line1) <= self.max_chars_per_line and len(new_line2) <= self.max_chars_per_line:
                return f"{new_line1}\n{new_line2}"
        return block_text

    def fix_orphan_blocks(self, blocks):
        """Fix blocks where 1-2 words from previous block appear in the next block"""
        if len(blocks) < 2:
            return blocks
        result = []
        i = 0
        while i < len(blocks) - 1:
            current_block = blocks[i]
            next_block = blocks[i + 1]
            current_words = current_block.replace('\n', ' ').split()
            next_words = next_block.replace('\n', ' ').split()
            if len(next_words) <= 2 and not self.is_sentence_end(' '.join(next_words)):
                all_words = current_words + next_words
                midpoint = len(all_words) // 2
                block1_text = ' '.join(all_words[:midpoint])
                block2_text = ' '.join(all_words[midpoint:])
                new_block1 = '\n'.join(self.split_long_line(block1_text))
                new_block2 = '\n'.join(self.split_long_line(block2_text))
                result.append(new_block1)
                result.append(new_block2)
                i += 2
                continue
            result.append(current_block)
            i += 1
        if i < len(blocks):
            result.append(blocks[i])
        return result

    def split_long_line(self, line, force_break_at_speaker=False):
        """Split a long line into multiple lines respecting max_chars_per_line"""
        if len(line) <= self.max_chars_per_line:
            return [line]
        if force_break_at_speaker:
            speaker_match = re.match(r'^([A-Za-z\s]+:)\s*(.*)', line)
            if speaker_match:
                speaker = speaker_match.group(1)
                text = speaker_match.group(2)
                if len(text) <= self.max_chars_per_line:
                    return [line]
                words = text.split()
                lines = []
                current_line = speaker
                for word in words:
                    if len(current_line + ' ' + word) <= self.max_chars_per_line:
                        current_line += ' ' + word
                    else:
                        lines.append(current_line)
                        current_line = word
                if current_line:
                    lines.append(current_line)
                return lines
        words = line.split()
        lines = []
        current_line = ''
        for word in words:
            if len(current_line + ' ' + word) <= self.max_chars_per_line:
                current_line += ' ' + word if current_line else word
            else:
                lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
        return lines

    def create_subtitle_block(self, index, start_time, end_time, text):
        """Create a properly formatted subtitle block"""
        if not text.strip():
            return None
        return f"{index}\n{start_time} --> {end_time}\n{text}"

    def is_sentence_end(self, text):
        """Return True if the text ends with a sentence-ending punctuation."""
        return any(text.strip().endswith(end) for end in self.sentence_endings)

    def parse_timecode(self, timecode):
        """Parse an SRT timecode string into a datetime object."""
        try:
            hours, minutes, seconds_ms = timecode.split(':')
            seconds, milliseconds = seconds_ms.split(',')
            return datetime(2000, 1, 1, int(hours), int(minutes), int(seconds), int(milliseconds) * 1000)
        except ValueError:
            logging.warning(f"Invalid timecode: {timecode}")
            return None

    def format_timecode(self, dt):
        """Format a datetime object into an SRT timecode string."""
        return dt.strftime('%H:%M:%S,%f')[:-3]

    def process_sentence_chunk(self, sentence_chunk, block_index, start_time, end_time):
        lines = []
        for block in sentence_chunk:
            text = block[3]
            block_lines = text.split('\n')
            for line in block_lines:
                if line.strip():
                    lines.append(line.strip())
        processed_lines = []
        for line in lines:
            for subline in line.split('/'):
                subline = subline.strip()
                if not subline:
                    continue
                split_lines = self.split_long_line(subline, force_break_at_speaker=True)
                processed_lines.extend(split_lines)
        subtitle_blocks = []
        current_block = []
        for line in processed_lines:
            if len(current_block) < self.max_lines_per_block:
                current_block.append(line)
            else:
                if len(current_block) == 2:
                    combined = ' '.join(current_block)
                    if len(combined) <= self.max_chars_per_line:
                        subtitle_blocks.append(combined)
                    else:
                        subtitle_blocks.append('\n'.join(current_block))
                else:
                    subtitle_blocks.append('\n'.join(current_block))
                current_block = [line]
        if current_block:
            if len(current_block) == 2:
                combined = ' '.join(current_block)
                if len(combined) <= self.max_chars_per_line:
                    subtitle_blocks.append(combined)
                else:
                    subtitle_blocks.append('\n'.join(current_block))
            else:
                subtitle_blocks.append('\n'.join(current_block))

        balanced_blocks = []
        for block in subtitle_blocks:
            lines = block.split('\n')
            text = ' '.join(lines)
            words = text.split()
            if len(words) < self.min_words_per_block and len(lines) == 1:
                balanced_blocks.append(block)
            elif len(lines) == 2:
                line1_words = lines[0].split()
                line2_words = lines[1].split()
                if len(line1_words) + len(line2_words) >= self.min_words_per_block:
                    total_words = len(line1_words) + len(line2_words)
                    target_words_per_line = total_words // 2
                    if abs(len(line1_words) - len(line2_words)) > 1:
                        new_line1 = ' '.join(words[:target_words_per_line])
                        new_line2 = ' '.join(words[target_words_per_line:])
                        balanced_blocks.append('\n'.join([new_line1, new_line2]))
                    else:
                        balanced_blocks.append(block)
                else:
                    balanced_blocks.append(block)
            else:
                balanced_blocks.append(block)
        fixed_blocks = []
        for block in balanced_blocks:
            words = block.replace('\n', ' ').split()
            if len(words) == 1 and fixed_blocks:
                fixed_blocks[-1] += ' ' + block
            else:
                fixed_blocks.append(block)
        balanced_blocks = fixed_blocks

        balanced_blocks = [self.balance_lines(block) for block in balanced_blocks]
        balanced_blocks = self.fix_orphan_blocks(balanced_blocks)

        start_dt = self.parse_timecode(start_time)
        end_dt = self.parse_timecode(end_time)
        total_duration = (end_dt - start_dt).total_seconds()
        if not balanced_blocks:
            return [], block_index
        total_chars = sum(len(block.replace('\n', ' ').replace(' ', '')) for block in balanced_blocks)
        if total_chars == 0:
            return [], block_index
        blocks = []
        current_time = start_dt
        for i, block_text in enumerate(balanced_blocks):
            block_chars = len(block_text.replace('\n', ' ').replace(' ', ''))
            block_duration = max(self.min_duration_short, min(self.max_duration, block_chars / self.cps))
            
            if i < len(balanced_blocks) - 1:
                block_end_dt = current_time + timedelta(seconds=block_duration)
                block_end = self.format_timecode(block_end_dt)
            else:
                # For the last block, ensure end_time is not earlier than start_time
                block_end_dt = current_time + timedelta(seconds=block_duration)
                if block_end_dt > end_dt:
                    block_end = end_time
                else:
                    block_end = self.format_timecode(block_end_dt)
                if self.parse_timecode(block_end) < current_time:
                    block_end = self.format_timecode(current_time + timedelta(seconds=self.min_duration_short))

            block_start = self.format_timecode(current_time)
            block = self.create_subtitle_block(block_index, block_start, block_end, block_text)
            if block:
                blocks.append(block)
            block_index += 1
            current_time = self.parse_timecode(block_end)
        return blocks, block_index

    def process_srt_in_chunks(self, input_file, output_file):
        """Process the SRT file in chunks to handle large files efficiently"""
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        blocks = re.split(r'\n\s*\n', content.strip())
        processed_blocks = []
        current_chunk = []
        block_index = 1
        for block in blocks:
            if not block.strip():
                continue
            lines = block.strip().split('\n')
            if len(lines) < 3:
                continue
            try:
                index = int(lines[0])
                timecode = lines[1]
                text = '\n'.join(lines[2:])
                start_time, end_time = timecode.split(' --> ')
                current_chunk.append((index, start_time, end_time, text))
                if len(current_chunk) >= self.chunk_size:
                    processed_chunk, block_index = self.process_sentence_chunk(
                        current_chunk, block_index, current_chunk[0][1], current_chunk[-1][2]
                    )
                    processed_blocks.extend(processed_chunk)
                    current_chunk = []
            except (ValueError, IndexError) as e:
                logging.warning(f"Error processing block: {e}")
                continue
        if current_chunk:
            processed_chunk, block_index = self.process_sentence_chunk(
                current_chunk, block_index, current_chunk[0][1], current_chunk[-1][2]
            )
            processed_blocks.extend(processed_chunk)
        with open(output_file, 'w', encoding='utf-8') as f:
            for block in processed_blocks:
                f.write(block + '\n\n')

    def recursive_format(self, input_file, output_file):
        temp_file = output_file + '.temp'
        current_file = input_file
        iteration = 0
        previous_hash = None
        while iteration < self.max_iterations:
            logging.info(f"Iteration {iteration + 1}")
            self.process_srt_in_chunks(current_file, temp_file)
            current_hash = hashlib.md5(open(temp_file, 'rb').read()).hexdigest()
            if current_hash == previous_hash:
                logging.info("No changes detected")
                break
            if current_file != input_file:
                os.remove(current_file)
            current_file = temp_file
            temp_file = output_file + f'.temp{iteration + 1}'
            previous_hash = current_hash
            iteration += 1
        shutil.move(current_file, output_file)
        logging.info(f"Final output: {output_file}")

def main():
    formatter = RecursiveSRTFormatter()
    input_file = r'C:\Users\WISE MARKET\Desktop\SRT_Formatting\input_srt_files\Pluscrew_76097_ED20TS_Bratty_Karens_Turn_Arrest_Into_a_Felony.srt'
    output_file = r'C:\Users\WISE MARKET\Desktop\SRT_Formatting\output_srt_files\formatted.srt'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    formatter.recursive_format(input_file, output_file)

if __name__ == '__main__':
    main()
