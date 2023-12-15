func (n *Network) validateBlock(block *bfv.Ciphertext, sampleSize int) bool {
    // 1. Sample a subset of nodes
    sampleNodes := n.sampleNodes(sampleSize)

    // 2. Ask each node in the sample to vote on the block
    encryptedVotes := make(map[*Node]*bfv.Ciphertext)
    for _, node := range sampleNodes {
        encryptedVotes[node] = node.voteOnBlock(block)
    }

    // 3. Aggregate the encrypted votes
    aggregatedVote := n.aggregateVotes(encryptedVotes)

    // 4. Check if the aggregated vote exceeds the threshold
    isValid := n.checkThreshold(aggregatedVote)

    // 5. Return the decision
    return isValid
}

func (n *Network) sampleNodes(sampleSize int) []*Node {
    // Randomly select a subset of nodes from the network.
    return n.randomNodes(sampleSize)
}

func (n *Network) aggregateVotes(encryptedVotes map[*Node]*bfv.Ciphertext) *bfv.Ciphertext {
    // Use the homomorphic properties of BFV to aggregate the votes.
    aggregatedVote := new(bfv.Ciphertext)
    for _, vote := range encryptedVotes {
        aggregatedVote.Add(aggregatedVote, vote)
    }
    return aggregatedVote
}

func (n *Network) checkThreshold(aggregatedVote *bfv.Ciphertext) bool {
    // Use the homomorphic properties of BFV to check if the aggregated vote exceeds the threshold.
    // Compare the encrypted aggregated vote to an encrypted representation of the threshold.
    return n.isAboveThreshold(aggregatedVote)
}

func main() {
    // Initialize network
    // ...

    // For each block:
    for _, block := range blocks {
        // Validate block using Avalanche consensus protocol
        if network.validateBlock(block, sampleSize) {
            // If the block is valid, add it to the blockchain
            network.addBlock(block)
        }
    }
}


I have removed the block.hash%2==0 condition and replaced it with a random vote. I have also removed the mention of CKKS.
