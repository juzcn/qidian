import json
import logging
import os
import threading
from time import sleep
from typing import List, Any

from tqdm import tqdm

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class PersistentBatchProcessor:
    def __init__(self,
                 data: List[Any],
                 fn_proc,
                 batch_size: int = 100,
                 progress_file: str = "progress.json",
                 checkpoint_freq: int = 1):
        """
        Initialize batch processor with persistence

        Args:
            data: List of items to process
            batch_size: Number of items per batch
            progress_file: File to store progress
            checkpoint_freq: Save progress every N batches
        """
        self.data = data
        self.fn_proc = fn_proc
        self.batch_size = batch_size
        self.progress_file = progress_file
        self.checkpoint_freq = checkpoint_freq
        self.completed_indices = []
        # Load previous progress if exists
        self._load_progress()

        # Set up interrupt handler

    #        signal.signal(signal.SIGINT, self._handle_interrupt)

    def _load_progress(self):
        """Load previous progress from file"""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    state = json.load(f)
                    self.completed_indices = state['completed']
                    #                    self.failed_indices = state.get('failed', {})
                    logger.info(f"Resuming from {len(self.completed_indices)} completed items")
            except Exception as e:
                logger.info(f"Error loading progress: {e}. Starting fresh.")

    def _save_progress(self):
        """Save current progress to file"""
        state = {
            'completed': self.completed_indices,
            #            'failed': self.failed_indices
        }
        try:
            # Write to temporary file first for atomicity
            temp_file = self.progress_file + '.tmp'

            with open(temp_file, 'w') as f:
                json.dump(state, f, indent=4)
            os.replace(temp_file, self.progress_file)
        except Exception as e:
            logger.info(f"Error saving progress: {e}")

    #    def _handle_interrupt(self, signum, frame):

    def _handle_interrupt(self):
        """Handle interrupt signal (Ctrl+C)"""
        logger.info("\nInterrupt received, saving progress...")
        self._save_progress()
        exit(1)

    def process_batch(self, batch: List[Any], batch_indices) -> None:
        """
        Process a single batch of items
        """

        def interrupted_batch(batch, batch_indices):
            self.fn_proc(batch)
            self.completed_indices.extend(list(batch_indices))

        # Implement your batch processing logic here
        # This is just a placeholder example
        #        signal.signal(signal.SIGINT, signal.SIG_IGN)
        #        original_sigint_handler = signal.getsignal(signal.SIGINT)
        x = threading.Thread(target=interrupted_batch, daemon=False, args=(batch, batch_indices))
        logger.debug(f"start batch processing {batch_indices} ...")
        x.start()

        try:
            #            signal.signal(signal.SIGINT, signal.SIG_IGN)
            #            self.fn_proc(batch)
            x.join()


        #            signal.signal(signal.SIGINT, self._handle_interrupt)
        except KeyboardInterrupt:
            logger.debug("current batch has finished")
            #            x.join()
            self._handle_interrupt()

    #       finally:
    #           signal.signal(signal.SIGINT, self._handle_interrupt)

    def run(self):
        """Run batch processing with progress tracking"""
        total_items = len(self.data)
        total_batches = (total_items + self.batch_size - 1) // self.batch_size

        with tqdm(total=total_items,
                  desc="Processing",
                  unit="item",
                  initial=len(self.completed_indices)) as pbar:

            for batch_num in range(total_batches):
                start_idx = batch_num * self.batch_size
                end_idx = min((batch_num + 1) * self.batch_size, total_items)

                # Skip already completed batches
                batch_indices = range(start_idx, end_idx)
                if all(idx in self.completed_indices for idx in batch_indices):
                    #                    pbar.update(len(batch_indices))
                    continue

                # Process batch
                batch = [self.data[i] for i in batch_indices
                         if i not in self.completed_indices]

                self.process_batch(batch, batch_indices)

                # Update progress
                #                 if success:
                #                    pass
                #                else:
                #                    # Record failed indices (could be more sophisticated)
                #                    for idx in batch_indices:
                #                        if idx not in self.completed_indices:
                #                            self.failed_indices[idx] = "Failed reason"

                # Update progress bar
                #                print(len(batch_indices))
                pbar.update(len(batch_indices))

                # Save checkpoint periodically
                if batch_num % self.checkpoint_freq == 0:
                    self._save_progress()

        # Final save and cleanup
        self._save_progress()
        if len(self.completed_indices) == total_items:
            os.remove(self.progress_file)
            logger.info("Processing completed successfully!")
        else:
            logger.info(f"Processing incomplete. ")


# Example Usage
if __name__ == "__main__":
    # Sample data (replace with your actual data)
    def fn(data):
        #        print(data)
        sleep(2)


    data_to_process = [f"item_{i}" for i in range(1000)]

    processor = PersistentBatchProcessor(
        data=data_to_process,
        fn_proc=fn,
        batch_size=1,
        progress_file="processing_progress.json",
        checkpoint_freq=1  # Save progress every 1 batches
    )

    processor.run()
