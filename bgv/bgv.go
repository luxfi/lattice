// Package bgv implements a unified RNS-accelerated version of the Fan-Vercauteren version of the Brakerski's scale invariant homomorphic encryption scheme (BFV) and Brakerski-Gentry-Vaikuntanathan (BGV) homomorphic encryption scheme. It provides modular arithmetic over the integers.
package bgv

import (
	"github.com/tuneinsight/lattigo/v4/rlwe"
)

// NewPlaintext allocates a new rlwe.Plaintext.
//
// inputs:
//   - params: an rlwe.GetRLWEParameters interface
//   - level: the level of the plaintext
//
// output: a newly allocated rlwe.Plaintext at the specified level.
//
// Note: the user can update the field `MetaData` to set a specific scaling factor,
// plaintext dimensions (if applicable) or encoding domain, before encoding values
// on the created plaintext.
func NewPlaintext(params Parameters, level int) (pt *rlwe.Plaintext) {
	pt = rlwe.NewPlaintext(params, level)
	pt.IsBatched = true
	pt.Scale = params.DefaultScale()
	pt.LogDimensions = params.LogMaxDimensions()
	return
}

// NewCiphertext allocates a new rlwe.Ciphertext.
//
// inputs:
//   - params: an rlwe.GetRLWEParameters interface
//   - degree: the degree of the ciphertext
//   - level: the level of the Ciphertext
//
// output: a newly allocated rlwe.Ciphertext of the specified degree and level.
func NewCiphertext(params Parameters, degree, level int) (ct *rlwe.Ciphertext) {
	ct = rlwe.NewCiphertext(params, degree, level)
	ct.IsBatched = true
	ct.Scale = params.DefaultScale()
	ct.LogDimensions = params.LogMaxDimensions()
	return
}

// NewEncryptor instantiates a new rlwe.Encryptor.
//
// inputs:
//   - params: an rlwe.GetRLWEParameters interface
//   - key: *rlwe.SecretKey or *rlwe.PublicKey
//
// output: an rlwe.Encryptor instantiated with the provided key.
func NewEncryptor(params Parameters, key rlwe.EncryptionKey) (*rlwe.Encryptor, error) {
	return rlwe.NewEncryptor(params, key)
}

// NewDecryptor instantiates a new rlwe.Decryptor.
//
// inputs:
//   - params: an rlwe.GetRLWEParameters interface
//   - key: *rlwe.SecretKey
//
// output: an rlwe.Decryptor instantiated with the provided key.
func NewDecryptor(params Parameters, key *rlwe.SecretKey) (*rlwe.Decryptor, error) {
	return rlwe.NewDecryptor(params, key)
}

// NewKeyGenerator instantiates a new rlwe.KeyGenerator.
//
// inputs:
//   - params: an rlwe.GetRLWEParameters interface
//
// output: an rlwe.KeyGenerator.
func NewKeyGenerator(params Parameters) *rlwe.KeyGenerator {
	return rlwe.NewKeyGenerator(params)
}
