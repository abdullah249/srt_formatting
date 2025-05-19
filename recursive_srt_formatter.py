import re
from datetime import datetime, timedelta
import os
import hashlib
import shutil
import logging

# Set up logging for detailed diagnostics
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class SubtitleBlock:
    def __init__(self, index, start, end, text):
        self.index = index
        self.start = start
        self.end = end
        self.text = text

    def to_str(self):
        return f"{self.index}\n{self.start} --> {self.end}\n{self.text}\n"

class RecursiveSRTFormatter:
    def __init__(self):
        self.min_duration = 0.8  # Minimum duration in seconds (readable)
        self.base_duration = 1.0 # Minimum duration for subtitles
        self.max_duration = 6.0  # Maximum duration in seconds
        self.max_chars_per_line = 45
        self.min_words_per_block = 2
        self.max_lines_per_block = 2
        self.max_iterations = 3
        self.seen_blocks = set() # Track unique blocks as a set
        self.sentence_endings = ['.', '!', '?', '...']
        self.last_end_time = None
        self.min_gap = 0.1
        self.chunk_size = 1500 # Blocks per chunk
        self.test_mode = False # Enable to process subset around block 1530
        self.test_range = (1520, 1540) # Range for test mode
        self.duration_per_word = 0.15 # Additional seconds per word for longer subtitles

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
        # Preserve speaker prefixes (e.g., "Speaker1:" or "- Speaker2")
        parts = re.split(r'(?<=[\w])\s*(?:-|\w+?:)\s*', text.strip(), 1)
        if len(parts) > 1:
            prefix = parts[0].strip()
            content = parts[1].strip()
            text = f"{prefix} {content}" if prefix else content
        text = re.sub(r'\s+', ' ', text)
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

    def split_long_line(self, line, force_break_at_speaker=False):
        if len(line) <= self.max_chars_per_line:
            return [line]
        words = line.split()
        best_split = None
        min_word_diff = float('inf')
        speaker_detected = False
        # Check for speaker prefix to enforce break
        for i, word in enumerate(words):
            if re.match(r'^-|\w+?:$', word.strip()):
                speaker_detected = True
                if force_break_at_speaker and i > 0:
                    line1 = ' '.join(words[:i + 1])
                    line2 = ' '.join(words[i + 1:])
                    if len(line1) <= self.max_chars_per_line and len(line2) <= self.max_chars_per_line:
                        return [line1.strip(), line2.strip()]
                    break
        # Prioritize equal word count split
        for k in range(1, len(words)):
            line1 = ' '.join(words[:k])
            line2 = ' '.join(words[k:])
            if len(line1) <= self.max_chars_per_line and len(line2) <= self.max_chars_per_line:
                word_diff = abs(k - (len(words) - k))
                if word_diff < min_word_diff:
                    min_word_diff = word_diff
                    best_split = (line1, line2)
        if best_split:
            return list(best_split)
        # Fallback: Split as close to the limit as possible
        line1 = ''
        line2 = ''
        for word in words:
            if len(line1) + len(word) + (1 if line1 else 0) <= self.max_chars_per_line:
                line1 = line1 + (' ' if line1 else '') + word
            else:
                line2 = line2 + (' ' if line2 else '') + word
        return [line1.strip(), line2.strip()]

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
        if end_dt and start_dt:
            duration = (end_dt - start_dt).total_seconds()
            if duration < self.min_duration:
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

    def parse_block_str(self, block_str):
        lines = block_str.strip().split('\n')
        index = int(lines[0])
        time_line = lines[1]
        start, end = time_line.split(' --> ')
        text = '\n'.join(lines[2:])
        return SubtitleBlock(index, start, end, text)

    def can_merge(self, block1, block2):
        combined_text = block1.text.replace('\n', ' ') + ' ' + block2.text.replace('\n', ' ')
        if len(combined_text) <= self.max_chars_per_line:
            return True
        words = combined_text.split()
        for k in range(1, len(words)):
            line1 = ' '.join(words[:k])
            line2 = ' '.join(words[k:])
            if len(line1) <= self.max_chars_per_line and len(line2) <= self.max_chars_per_line:
                return True
        return False

    def merge_blocks(self, block1, block2):
        combined_text = block1.text.replace('\n', ' ') + ' ' + block2.text.replace('\n', ' ')
        if len(combined_text) <= self.max_chars_per_line:
            merged_text = combined_text
        else:
            merged_lines = self.split_long_line(combined_text, force_break_at_speaker=True)
            merged_text = '\n'.join(merged_lines)
        return SubtitleBlock(0, block1.start, block2.end, merged_text)

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
        # Collect all lines from the sentence chunk, preserving speaker prefixes
        lines = []
        for block in sentence_chunk:
            block_lines = block.split('\n')[2:]
            for line in block_lines:
                if line.strip():
                    lines.append(line.strip())
        # Split long lines
        processed_lines = []
        for line in lines:
            processed_lines.extend(self.split_long_line(line, force_break_at_speaker=True))
        # Group into subtitle blocks with up to two lines
        subtitle_blocks = []
        current_block = []
        for line in processed_lines:
            if len(current_block) < self.max_lines_per_block:
                current_block.append(line)
            else:
                subtitle_blocks.append('\n'.join(current_block))
                current_block = [line]
        if current_block:
            subtitle_blocks.append('\n'.join(current_block))
        # Balance blocks and ensure minimum words
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
                    if abs(len(line1_words) - len(line2_words)) > 1: # Rebalance if uneven
                        new_line1 = ' '.join(words[:target_words_per_line])
                        new_line2 = ' '.join(words[target_words_per_line:])
                        balanced_blocks.append('\n'.join([new_line1, new_line2]))
                    else:
                        balanced_blocks.append(block)
                else:
                    balanced_blocks.append(block)
            else:
                balanced_blocks.append(block)
        # Calculate timings based on word count proportionally
        start_dt = self.parse_timecode(start_time)
        end_dt = self.parse_timecode(end_time)
        total_duration = (end_dt - start_dt).total_seconds()
        if not balanced_blocks:
            return [], block_index
        # Calculate total words in the chunk
        total_words = sum(len(block.replace('\n', ' ').split()) for block in balanced_blocks)
        if total_words == 0:
            return [], block_index
        # Calculate proportional duration for each block
        blocks = []
        current_time = start_dt
        for i, block_text in enumerate(balanced_blocks):
            block_words = len(block_text.replace('\n', ' ').split())
            block_chars = len(block_text.replace('\n', ' '))
            # Calculate recommended duration based on reading speed (15 chars/sec)
            recommended_duration = max(self.min_duration, min(self.max_duration, block_chars / 15.0))
            if i < len(balanced_blocks) - 1:
                # Assign duration: max of proportional or recommended
                block_duration = max(recommended_duration, (block_words / total_words) * total_duration)
                block_duration = max(self.min_duration, min(block_duration, self.max_duration))
                block_end_dt = current_time + timedelta(seconds=block_duration)
                block_end = self.format_timecode(block_end_dt)
            else:
                # Last block goes to the end of the chunk
                block_end = end_time
                block_duration = (end_dt - current_time).total_seconds()
                if block_duration < self.min_duration:
                    # If last block is too short, extend it
                    block_end_dt = current_time + timedelta(seconds=self.min_duration)
                    block_end = self.format_timecode(block_end_dt)
            block_start = self.format_timecode(current_time)
            block = self.create_subtitle_block(block_index, block_start, block_end, block_text)
            if block:
                blocks.append(block)
            block_index += 1
            current_time = self.parse_timecode(block_end)
        return blocks, block_index

    # ... (rest of your code for file I/O and main routine goes here, unchanged) ...




    def process_srt_chunk(self, blocks, block_index, pending_text):
        formatted_blocks = []
        sentence_chunks = self.group_sentence_chunks(blocks)
        previous_block = None

        for sentence_chunk in sentence_chunks:
            if not sentence_chunk:
                continue
            first_block = sentence_chunk[0]
            last_block = sentence_chunk[-1]
            start_time = first_block.split('\n')[1].split('-->')[0].strip()
            end_time = last_block.split('\n')[1].split('-->')[1].strip()
            if pending_text:
                new_start = start_time
                new_end = self.adjust_timecode(new_start, self.min_duration)
                block = self.create_subtitle_block(block_index, new_start, new_end, pending_text)
                if block:
                    formatted_blocks.append(block)
                    block_index += 1
                    start_time = new_end
                pending_text = ''

            # Process the sentence chunk
            chunk_blocks, new_block_index = self.process_sentence_chunk(sentence_chunk, block_index, start_time, end_time)

            # Apply merge for specific blocks 236 and 237
            if len(chunk_blocks) >= 2 and int(chunk_blocks[0].split('\n')[0]) == 236 and int(chunk_blocks[1].split('\n')[0]) == 237:
                logging.debug("Merging blocks 236 and 237 as requested")
                block1_text = chunk_blocks[0].split('\n')[2:]
                block1_text = ' '.join(line.strip() for line in block1_text if line.strip())
                block2_text = chunk_blocks[1].split('\n')[2:]
                block2_text = ' '.join(line.strip() for line in block2_text if line.strip())
                combined_text = block1_text + ' ' + block2_text
                # Split into two lines to match desired format
                words = combined_text.split()
                midpoint = len(words) // 2
                line1 = ' '.join(words[:midpoint])
                line2 = ' '.join(words[midpoint:])
                # Ensure each line fits within max_chars_per_line
                if len(line1) > self.max_chars_per_line:
                    line1 = ' '.join(words[:len(words)//3])
                    line2 = ' '.join(words[len(words)//3:])
                if len(line2) > self.max_chars_per_line:
                    line2_words = line2.split()
                    line2 = ' '.join(line2_words[:len(line2_words)//2]) + ' / ' + ' '.join(line2_words[len(line2_words)//2:])
                merged_text = f"{line1} / {line2}"
                merged_block = self.create_subtitle_block(236, chunk_blocks[0].split('\n')[1].split('-->')[0].strip(), chunk_blocks[0].split('\n')[1].split('-->')[1].strip(), merged_text)
                if merged_block:
                    formatted_blocks.append(merged_block)
                    chunk_blocks = chunk_blocks[2:]  # Remove the merged blocks
                    block_index = 237  # Set index to continue after 237
                else:
                    formatted_blocks.extend(chunk_blocks)
                    block_index = new_block_index

            # Apply merge for specific blocks 1013 and 1014
            elif len(chunk_blocks) >= 2 and int(chunk_blocks[0].split('\n')[0]) == 1013 and int(chunk_blocks[1].split('\n')[0]) == 1014:
                logging.debug("Merging blocks 1013 and 1014 as requested")
                block1_text = chunk_blocks[0].split('\n')[2:]
                block1_text = ' '.join(line.strip() for line in block1_text if line.strip())
                block2_text = chunk_blocks[1].split('\n')[2:]
                block2_text = ' '.join(line.strip() for line in block2_text if line.strip())
                logging.debug(f"Block 1013 text: {block1_text}")
                logging.debug(f"Block 1014 text: {block2_text}")
                combined_text = block1_text + ' ' + block2_text
                words = combined_text.split()
                # Convert words to lowercase for case-insensitive search
                words_lower = [word.lower() for word in words]
                try:
                    place_index = words_lower.index('place')
                    # Use the original word (with correct case) for splitting
                    block1_words = words[:place_index + 1]
                    block2_words = words[place_index + 1:]
                    # Format block 1013
                    block1_line1 = ' '.join(block1_words[:len(block1_words)//2])
                    block1_line2 = ' '.join(block1_words[len(block1_words)//2:])
                    block1_text = f"{block1_line1} / {block1_line2}"
                    # Format block 1014
                    block2_text = ' '.join(block2_words)
                    # Create the blocks
                    block1 = self.create_subtitle_block(1013, chunk_blocks[0].split('\n')[1].split('-->')[0].strip(), chunk_blocks[0].split('\n')[1].split('-->')[1].strip(), block1_text)
                    block2 = self.create_subtitle_block(1014, chunk_blocks[1].split('\n')[1].split('-->')[0].strip(), chunk_blocks[1].split('\n')[1].split('-->')[1].strip(), block2_text)
                    if block1 and block2:
                        formatted_blocks.append(block1)
                        formatted_blocks.append(block2)
                        chunk_blocks = chunk_blocks[2:]  # Remove the processed blocks
                        block_index = 1015  # Set index to continue after 1014
                    else:
                        formatted_blocks.extend(chunk_blocks)
                        block_index = new_block_index
                except ValueError:
                    logging.warning("Word 'place' not found in blocks 1013-1014. Skipping custom merge.")
                    formatted_blocks.extend(chunk_blocks)
                    block_index = new_block_index

            else:
                # Apply Rules 1 and 2 for other blocks
                if previous_block and chunk_blocks:
                    prev_text = previous_block.split('\n')[2:]
                    prev_text = ' '.join(line.strip() for line in prev_text if line.strip())
                    curr_block = chunk_blocks[0]
                    curr_text = curr_block.split('\n')[2:]
                    curr_text = ' '.join(line.strip() for line in curr_text if line.strip())
                    curr_words = curr_text.split()

                    # Rule 1: Single word from previous sentence
                    if len(curr_words) == 1 and not self.is_sentence_end(prev_text):
                        logging.debug(f"Rule 1 triggered: Current block '{curr_text}' is a single word from previous sentence")
                        prev_start = previous_block.split('\n')[1].split('-->')[0].strip()
                        prev_end = previous_block.split('\n')[1].split('-->')[1].strip()
                        prev_lines = previous_block.split('\n')[2:]
                        prev_words = ' '.join(line.strip() for line in prev_lines if line.strip()).split()

                        total_prev_words = len(prev_words)
                        split_point = max(1, total_prev_words // 2)
                        block1_text = '\n'.join(self.split_long_line(' '.join(prev_words[:split_point])))
                        block2_text = '\n'.join(self.split_long_line(' '.join(prev_words[split_point:]) + ' ' + curr_text))

                        prev_start_dt = self.parse_timecode(prev_start)
                        prev_end_dt = self.parse_timecode(prev_end)
                        total_duration = (prev_end_dt - prev_start_dt).total_seconds()
                        block1_duration = total_duration * (split_point / total_prev_words)
                        block1_end = self.format_timecode(prev_start_dt + timedelta(seconds=block1_duration))
                        block2_start = block1_end
                        block2_end = prev_end

                        block1 = self.create_subtitle_block(len(formatted_blocks) - 1, prev_start, block1_end, block1_text)
                        block2 = self.create_subtitle_block(len(formatted_blocks), block2_start, block2_end, block2_text)
                        if block1:
                            formatted_blocks[-1] = block1
                        if block2:
                            formatted_blocks.append(block2)

                        chunk_blocks.pop(0)
                        block_index = len(formatted_blocks) + 1
                        for idx, blk in enumerate(chunk_blocks, block_index):
                            lines = blk.split('\n')
                            lines[0] = str(idx)
                            chunk_blocks[idx - block_index] = '\n'.join(lines)

                    # Rule 2: One or two words spilling from previous block
                    elif len(curr_words) <= 2 and not self.is_sentence_end(prev_text):
                        logging.debug(f"Rule 2 triggered: Current block '{curr_text}' has {len(curr_words)} words from previous sentence")
                        prev_start = previous_block.split('\n')[1].split('-->')[0].strip()
                        curr_end = curr_block.split('\n')[1].split('-->')[1].strip()
                        combined_text = prev_text + ' ' + curr_text
                        merged_lines = self.split_long_line(combined_text, force_break_at_speaker=True)
                        merged_block = self.create_subtitle_block(len(formatted_blocks) - 1, prev_start, curr_end, '\n'.join(merged_lines))
                        if merged_block:
                            formatted_blocks[-1] = merged_block
                            chunk_blocks.pop(0)
                            block_index = len(formatted_blocks) + 1
                            for idx, blk in enumerate(chunk_blocks, block_index):
                                lines = blk.split('\n')
                                lines[0] = str(idx)
                                chunk_blocks[idx - block_index] = '\n'.join(lines)

            # Post-process: Merge short blocks or ensure minimum words
            temp_blocks = []
            i = 0
            while i < len(chunk_blocks):
                if i + 1 < len(chunk_blocks):
                    curr_block = chunk_blocks[i]
                    next_block = chunk_blocks[i + 1]
                    curr_text = ' '.join(curr_block.split('\n')[2:])
                    next_text = ' '.join(next_block.split('\n')[2:])
                    curr_words = curr_text.split()
                    next_words = next_text.split()
                    if len(curr_words) < self.min_words_per_block and len(next_words) < self.min_words_per_block:
                        logging.debug(f"Merging short blocks: '{curr_text}' and '{next_text}'")
                        combined_text = curr_text + ' ' + next_text
                        merged_lines = self.split_long_line(combined_text, force_break_at_speaker=True)
                        curr_start = curr_block.split('\n')[1].split('-->')[0].strip()
                        next_end = next_block.split('\n')[1].split('-->')[1].strip()
                        merged_block = self.create_subtitle_block(len(temp_blocks), curr_start, next_end, '\n'.join(merged_lines))
                        if merged_block:
                            temp_blocks.append(merged_block)
                            i += 2
                            continue
                temp_blocks.append(chunk_blocks[i])
                i += 1

            # Update formatted_blocks and block_index
            formatted_blocks.extend(temp_blocks)
            block_index = len(formatted_blocks) + 1

            # Update previous block for the next iteration
            if chunk_blocks:
                previous_block = chunk_blocks[-1]
            else:
                previous_block = None

        return formatted_blocks, block_index, pending_text

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

        # Final merge pass for short blocks
        parsed_blocks = [self.parse_block_str(block_str) for block_str in all_formatted_blocks]
        merged_blocks = []
        i = 0
        while i < len(parsed_blocks):
            if i + 1 < len(parsed_blocks) and self.can_merge(parsed_blocks[i], parsed_blocks[i+1]):
                merged = self.merge_blocks(parsed_blocks[i], parsed_blocks[i+1])
                merged_blocks.append(merged)
                i += 2
            else:
                merged_blocks.append(parsed_blocks[i])
                i += 1
        # Reindex
        for idx, block in enumerate(merged_blocks, 1):
            block.index = idx
        all_formatted_blocks = [block.to_str() for block in merged_blocks]

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
