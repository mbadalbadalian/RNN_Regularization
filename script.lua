require 'torch'
print(torch)

local dataset = require("data")  -- Load the dataset module

-- local train_data = dataset.traindataset(20)  -- Call the function with batch_size = 10
local valid_data, vocab_map = dataset.validdataset(20)
-- local test_data = dataset.testdataset(20)

-- Function to print first 10 entries
local function print_first_n_entries(data, n, vocab_map)
    local data_size = data:size(1)  -- Extract first dimension size
    print("Dataset size:", data_size)  -- Debugging output

    for i = 1, math.min(n, data_size) do
        print("Entry " .. i .. ":")
        print(data[i])
        for i = 1, 20 do
            print(vocab_map[data[i]])
        end
    end
end

print("\nValidation Data (First 10 Entries):")
print_first_n_entries(valid_data, 10, vocab_map)
