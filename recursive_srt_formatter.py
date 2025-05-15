import re
from datetime import datetime, timedelta
import os
import hashlib
import shutil
import logging

# Set up logging for detailed diagnostics
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class RecursiveSRTFormatter:
    def __init__(self):
        self.min_duration = 1.0  # Minimum duration in seconds
        self.max_duration = 6.0  # Maximum duration in seconds
        self.max_chars_per_line = 45
        self.min_words_per_block = 4
        self.max_lines_per_block = 2
        self.max_iterations = 3
        self.seen_blocks = set()  # Track unique blocks as a set
        self.sentence_endings = ['.', '!', '?', '...']
        self.last_end_time = None
        self.min_gap = 0.1
        self.chunk_size = 1500  # Blocks per chunk
        self.test_mode = False  # Enable to process subset around block 1530
        self.test_range = (1520, 1540)  # Range for test mode

    def parse_timecode(self, timecode):
        try:
            hours, minutes, seconds_ms = timecode.split(':')
            seconds, milliseconds = seconds_ms.split(',')
            return datetime(2000, 1, 1, int(hours), int(minutes), int(seconds), int(milliseconds) * 1000)
        except ValueError:
            logging.warning(f"Invalid timecode: {timecode}")
            return None

    def format_timecode(self, dt):
        return dt.strftime('%H:%M:%S,%f')[:-3]

    def adjust_timecode(self, timecode, seconds):
        dt = self.parse_timecode(timecode)
        if dt:
            dt += timedelta(seconds=seconds)
            return self.format_timecode(dt)
        return timecode

    def clean_text(self, text):
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'^-+\s*', '- ', text)
        return text.strip()

    def is_duplicate_block(self, text, start_time, end_time):
        block_hash = hashlib.md5(f"{text}|{start_time}|{end_time}".encode()).hexdigest()
        if block_hash in self.seen_blocks:
            logging.debug(f"Duplicate block detected: {block_hash}")
            return True
        self.seen_blocks.add(block_hash)
        return False

    def is_sentence_end(self, text):
        return any(text.strip().endswith(end) for end in self.sentence_endings)

    def split_text_into_subtitle_blocks(self, text):
        words = text.split()
        lines = []
        current_line = []
        current_length = 0

        # Build lines
        for word in words:
            word_len = len(word) + (1 if current_line else 0)
            if current_length + word_len <= self.max_chars_per_line:
                current_line.append(word)
                current_length += word_len
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)

        if current_line:
            lines.append(' '.join(current_line))

        # Adjust if last line has only one word
        while len(lines) > 1 and len(lines[-1].split()) == 1:
            prev_words = lines[-2].split()
            if len(prev_words) > 1:
                last_word_prev = prev_words[-1]
                new_last_line = last_word_prev + ' ' + lines[-1]
                if len(new_last_line) <= self.max_chars_per_line:
                    lines[-2] = ' '.join(prev_words[:-1])
                    lines[-1] = new_last_line
                    if not lines[-2]:  # Remove empty line
                        lines.pop(-2)
                else:
                    break
            else:
                break

        # Group lines into blocks
        subtitle_blocks = []
        temp_block = []
        for line in lines:
            if len(temp_block) < self.max_lines_per_block:
                temp_block.append(line)
            else:
                subtitle_blocks.append('\n'.join(temp_block))
                temp_block = [line]
        if temp_block:
            subtitle_blocks.append('\n'.join(temp_block))

        # Post-process: if last block has one word, split previous block
        final_blocks = []
        for i, block in enumerate(subtitle_blocks):
            block_lines = block.split('\n')
            if i == len(subtitle_blocks) - 1 and len(block_lines) == 1 and len(block_lines[0].split()) == 1:
                if i > 0:
                    # Split the previous block
                    prev_block = subtitle_blocks[i - 1]
                    prev_lines = prev_block.split('\n')
                    if len(prev_lines) == 2:
                        # Move second line of previous block to join the single word
                        new_block1 = prev_lines[0]
                        new_block2 = prev_lines[1] + ' ' + block_lines[0]
                        if len(new_block2) <= self.max_chars_per_line:
                            final_blocks[-1] = new_block1  # Replace previous block
                            final_blocks.append(new_block2)
                        else:
                            # Split previous block differently
                            prev_words = ' '.join(prev_lines).split()
                            mid = len(prev_words) // 2
                            block1_words = prev_words[:mid]
                            block2_words = prev_words[mid:] + block_lines[0].split()
                            block1 = ' '.join(block1_words)
                            block2 = ' '.join(block2_words)
                            if len(block1) <= self.max_chars_per_line and len(block2) <= self.max_chars_per_line:
                                final_blocks[-1] = block1
                                final_blocks.append(block2)
                            else:
                                final_blocks.append(block)  # Fallback
                    else:
                        final_blocks.append(block)  # Previous block has 1 line, append as is
                else:
                    final_blocks.append(block)  # No previous block to split
            else:
                final_blocks.append(block)

        logging.debug(f"Split text into blocks: {final_blocks}")
        return final_blocks

    def validate_timecode(self, timecode):
        try:
            time_part, ms_part = timecode.split(',')
            hours, minutes, seconds = map(int, time_part.split(':'))
            milliseconds = int(ms_part)
            if 0 <= hours <= 23 and 0 <= minutes <= 59 and 0 <= seconds <= 59 and 0 <= milliseconds <= 999:
                return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
            return None
        except ValueError:
            logging.warning(f"Invalid timecode: {timecode}")
            return None

    def fix_timecode_overlap(self, start_time, end_time, last_end_time):
        start_dt = self.parse_timecode(start_time)
        end_dt = self.parse_timecode(end_time)
        if last_end_time:
            last_end_dt = self.parse_timecode(last_end_time)
            if start_dt and last_end_dt and start_dt < last_end_dt:
                start_dt = last_end_dt + timedelta(seconds=self.min_gap)
                start_time = self.format_timecode(start_dt)
        if end_dt and start_dt and end_dt <= start_dt:
            end_dt = start_dt + timedelta(seconds=self.min_duration)
            end_time = self.format_timecode(end_dt)
        return start_time, end_time

    def create_subtitle_block(self, index, start_time, end_time, text):
        start_time = self.validate_timecode(start_time)
        end_time = self.validate_timecode(end_time)
        if not (start_time and end_time):
            logging.warning(f"Invalid timecodes for block {index}")
            return None
        start_time, end_time = self.fix_timecode_overlap(start_time, end_time, self.last_end_time)
        self.last_end_time = end_time
        block = f"{index}\n{start_time} --> {end_time}\n{text}\n"
        if self.is_duplicate_block(text, start_time, end_time):
            logging.warning(f"Skipping duplicate block at {start_time} --> {end_time}")
            return None
        return block

    def group_sentence_chunks(self, blocks):
        sentence_chunks = []
        current_chunk = []
        for block in blocks:
            lines = block.split('\n')
            text = ' '.join(lines[2:]).strip()
            current_chunk.append(block)
            if self.is_sentence_end(text):
                sentence_chunks.append(current_chunk)
                current_chunk = []
        if current_chunk:
            sentence_chunks.append(current_chunk)
        return sentence_chunks

    def process_sentence_chunk(self, sentence_chunk, block_index, start_time, end_time):
        full_text = ' '.join([' '.join(block.split('\n')[2:]).strip() for block in sentence_chunk])
        subtitle_blocks = self.split_text_into_subtitle_blocks(full_text)
        if not subtitle_blocks:
            return [], ''

        start_dt = self.parse_timecode(start_time)
        end_dt = self.parse_timecode(end_time)
        total_duration = (end_dt - start_dt).total_seconds()
        num_blocks = len(subtitle_blocks)
        duration_per_block = total_duration / num_blocks if num_blocks > 0 else 0

        formatted_blocks = []
        current_time = start_dt
        for block_text in subtitle_blocks:
            block_start = self.format_timecode(current_time)
            current_time += timedelta(seconds=duration_per_block)
            block_end = self.format_timecode(current_time)
            block = self.create_subtitle_block(block_index, block_start, block_end, block_text)
            if block:
                formatted_blocks.append(block)
                block_index += 1

        return formatted_blocks, ''

    def process_srt_chunk(self, blocks, block_index, pending_text):
        formatted_blocks = []
        sentence_chunks = self.group_sentence_chunks(blocks)
        for sentence_chunk in sentence_chunks:
            if not sentence_chunk:
                continue
            first_block = sentence_chunk[0]
            last_block = sentence_chunk[-1]
            start_time = first_block.split('\n')[1].split('-->')[0].strip()
            end_time = last_block.split('\n')[1].split('-->')[1].strip()
            chunk_text = ' '.join([' '.join(block.split('\n')[2:]).strip() for block in sentence_chunk])
            if pending_text:
                chunk_text = f"{pending_text} {chunk_text}".strip()
                pending_text = ''
            chunk_formatted, excess_text = self.process_sentence_chunk(sentence_chunk, block_index, start_time, end_time)
            formatted_blocks.extend(chunk_formatted)
            block_index += len(chunk_formatted)
            if excess_text:
                logging.warning(f"Excess text after processing sentence chunk: {excess_text}")
                pending_text = excess_text

        if pending_text:
            last_end_time = formatted_blocks[-1].split('\n')[1].split('-->')[1].strip() if formatted_blocks else '00:00:00,000'
            new_start = self.adjust_timecode(last_end_time, self.min_gap)
            new_end = self.adjust_timecode(new_start, self.min_duration)
            block = self.create_subtitle_block(block_index, new_start, new_end, pending_text)
            if block:
                formatted_blocks.append(block)
                block_index += 1

        return formatted_blocks, block_index, ''

    def parse_srt_blocks(self, content):
        blocks = []
        current_block = []
        for line in content.splitlines():
            line = line.strip()
            if line.isdigit() and not current_block:
                current_block.append(line)
            elif '-->' in line and len(current_block) == 1:
                current_block.append(line)
            elif current_block and len(current_block) >= 2:
                current_block.append(line)
            if line == '' and current_block:
                blocks.append('\n'.join(current_block))
                current_block = []
        if current_block:
            blocks.append('\n'.join(current_block))
        return [block for block in blocks if block.strip()]

    def process_srt_in_chunks(self, input_file, output_file):
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logging.error(f"Failed to read {input_file}: {e}")
            return
        self.seen_blocks.clear()
        self.last_end_time = None
        blocks = self.parse_srt_blocks(content)
        if not blocks:
            logging.error("No valid blocks found")
            return
        if self.test_mode:
            logging.info(f"Test mode: Processing blocks {self.test_range[0]} to {self.test_range[1]}")
        chunked_blocks = [blocks[i:i + self.chunk_size] for i in range(0, len(blocks), self.chunk_size)]
        block_index = 1
        pending_text = ''
        all_formatted_blocks = []
        for chunk_idx, chunk in enumerate(chunked_blocks):
            logging.info(f"Processing chunk {chunk_idx + 1}/{len(chunked_blocks)}")
            formatted, block_index, pending_text = self.process_srt_chunk(chunk, block_index, pending_text)
            all_formatted_blocks.extend(formatted)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(all_formatted_blocks))
        logging.info(f"Output written to {output_file}")

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
