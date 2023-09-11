## TODO List

#### Operation Definitions

Define the following operations
1. Encoding Operations
   1. Patch Partition
      1. Patch Partition needs to be resized for Linear Embedding
   2. Linear Embedding
   3. Position Embedding
      1. Need to define a sinusoid function to take shape - Might just use Attention is all you need's function
   4. Window Masking -- Couple of potential implementations
      1. Removing masked
      2. Setting masked to 0
   5. Patch merging
      1. 
2. Decoding Operations
   1. Decoder Embedding
   2. Position Embedding
   3. Layer Normalization
   4. Patch Expanding
   5. Predictor Projection

Each of these operations should have a dedicated function. Let's start it with a ```.ipynb``` document.
Further, all of them should be accompanied by shaping specifications

#### Block Definitions

We additionally need to define SWIN Transformer blocks, these will also change shapes. Refer to [Hugging Face]()

