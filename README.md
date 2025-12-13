# Lattice: lattice-based multiparty homomorphic encryption library in Go

<p align="center">
	<img src="logo.png" />
</p>

![Go tests](https://github.com/luxfi/lattice/actions/workflows/ci.yml/badge.svg)

Lattice is a Go module that implements full-RNS Ring-Learning-With-Errors-based homomorphic-encryption
primitives and Multiparty-Homomorphic-Encryption-based secure protocols. The library features:

- Optimized arithmetic for power-of-two cyclotomic rings.
- Advanced and scheme-agnostic implementation of RLWE-based primitives, key-generation, and their multiparty version.
- Implementation of the BFV/BGV and CKKS schemes and their multiparty version.
- Support for RGSW, external product and LMKCDEY blind rotations.
- A pure Go implementation, enabling cross-platform builds, including WASM compilation for
browser clients, with comparable performance to state-of-the-art C++ libraries.

Lattice is meant to support HE in distributed systems and microservices architectures, for which Go
is a common choice thanks to its natural concurrency model and portability.

## Library overview

<p align="center" width="100%">
  <img width=500 height=350 alt="lattice-hierarchy" src="./lattice-hierarchy.svg">
</p>

Lattice is a strictly hierarchical library whose packages form a linear dependency chain ranging from
low-level arithmetic functionalities to high-level homomorphic circuits. A graphical depiction of the
Lattice package organization is given in the Figure above.

- `lattice/ring`: At the lowest level resides the `ring` package providing modular arithmetic operations for polynomials
  in the RNS basis, including: RNS basis extension; RNS rescaling; number theoretic transform (NTT); uniform,
  Gaussian and ternary sampling.

  - `lattice/core`: This package implements the core cryptographic functionalities of the library and builds directly
    upon the arithmetic functionalities provided by the `ring` package:

	- `rlwe`: Common base for generic RLWE-based homomorphic encryption.
      It provides all homomorphic functionalities and defines all structs that are not scheme-specific.
      This includes plaintext, ciphertext, key-generation, encryption, decryption and key-switching, as
      well as other more advanced primitives such as RLWE-repacking.

    - `rgsw`: A Full-RNS variant of Ring-GSW ciphertexts and the external product.

- `lattice/schemes`: The implementation of RLWE-based homomorphic encryption schemes are found in the `schemes` package:

  - `bfv`: A Full-RNS variant of the Brakerski-Fan-Vercauteren scale-invariant homomorphic
    encryption scheme. This scheme is instantiated via a wrapper of the `bgv` scheme.
    It provides modular arithmetic over the integers.

  - `bgv`: A Full-RNS generalization of the Brakerski-Fan-Vercauteren scale-invariant (BFV) and
    Brakerski-Gentry-Vaikuntanathan (BGV) homomorphic encryption schemes.
    It provides modular arithmetic over the integers.

  - `ckks`: A Full-RNS Homomorphic Encryption for Arithmetic for Approximate Numbers (HEAAN,
    a.k.a. CKKS) scheme. It provides fixed-point approximate arithmetic over the complex numbers (in its classic
    variant) and over the real numbers (in its conjugate-invariant variant).

- `lattice/circuits`: The circuits package provides implementation of a select set of homomorphic circuits for
  the `bgv` and `ckks` cryptosystems:

  - `bgv/lintrans`, `ckks/lintrans`: Arbitrary linear transformations and slot permutations for both `bgv` and `ckks`.
    Scheme-generic objects and functions are part of `common/lintrans`.

  - `bgv/polynomial`, `ckks/polynomial`: Polynomial evaluation circuits for `bgv` and `ckks`.
    Scheme-generic objects and functions are part of `common/polynomial`.

  - `ckks/minimax`: Minimax composite polynomial evaluator for `ckks`.

  - `ckks/comparison`: Homomorphic comparison-based circuits such as `sign`, `max` and `step` for the `ckks` scheme.

  - `ckks/inverse`: Homomorphic inverse circuit for `ckks`.

  - `ckks/mod1`: Homomorphic circuit for the `mod1` function using the `ckks` cryptosystem.

  - `ckks/dft`: Homomorphic Discrete Fourier Transform circuits for the `ckks` scheme.

  - `ckks/bootstrapping`: Bootstrapping for fixed-point approximate arithmetic over the real
     and complex numbers, i.e., the `ckks` scheme, with support for the Conjugate Invariant ring, batch bootstrapping with automatic
     packing/unpacking of sparsely packed/smaller ring degree ciphertexts, arbitrary precision bootstrapping,
     and advanced circuit customization/parameterization.

- `lattice/multiparty`: Package for multiparty (a.k.a. distributed or threshold) key-generation and
  interactive ciphertext bootstrapping with secret-shared secret keys.

  - `mpckks`: Homomorphic decryption and re-encryption from and to Linear-Secret-Sharing-Shares,
    as well as interactive ciphertext bootstrapping for the `schemes/ckks` package.

  - `mpbgv`: Homomorphic decryption and re-encryption from and to Linear-Secret-Sharing-Shares,
    as well as interactive ciphertext bootstrapping for the `schemes/bgv` package.

- `lattice/examples`: Executable Go programs that demonstrate the use of the Lattice library. Each
                      subpackage includes test files that further demonstrate the use of Lattice
                      primitives.

- `lattice/utils`: Generic utility methods. This package also contains the following sub-packages:
  - `bignum`: Arbitrary precision linear algebra and polynomial approximation.
  - `buffer`: Efficient methods to write/read on `io.Writer` and `io.Reader`.
  - `factorization`: Various factorization algorithms for medium-sized integers.
  - `sampling`: Secure bytes sampling.
  - `structs`: Generic structs for maps, vectors and matrices, including serialization.

### Documentation

The full documentation of the individual packages can be browsed as a web page using official
Golang documentation rendering tool `pkgsite` or browsing the [Go doc](https://pkg.go.dev/github.com/luxfi/lattice/v6).

```bash
$ go install golang.org/x/pkgsite/cmd/pkgsite@latest
$ cd lattice
$ pkgsite -open .
```

## Versions and Roadmap

The Lattice library was originally exclusively developed by the EPFL Laboratory for Data Security
until its version 2.4.0.

Starting with the release of version 3.0.0, Lattice is maintained and supported
by [Lux Industries Inc(https://lux.network).

Also starting from version 3.0.0, the module name has changed to
`github.com/luxfi/lattice/v[X]`, and the official repository has been moved to
https://github.com/luxfi/lattice. This has the following implications for modules that depend
on Lattice:
- Modules that require `github.com/ldsec/lattice/v2` will still build correctly.
- To upgrade to a version X.y.z >= 3.0.0, depending modules must require `github.com/luxfi/lattice/v[X]/`,
  for example by changing the imports to `github.com/luxfi/lattice/v[X]/[package]` and by
  running `go mod tidy`.

The current version of Lattice (v6.x.x) is fast-evolving and in constant development. Consequently,
there will still be backward-incompatible changes within this major version, in addition to many bug
fixes and new features. Hence, we encourage all Lattice users to update to the latest Lattice version.

See CHANGELOG.md for the current and past versions.

## Pull Requests

External pull requests should only be used to propose new functionalities that are substantial and would
require a fair amount of work if done on our side. If you plan to open such a pull request, please contact
us before doing so to make sure that the proposed changes are aligned with our development roadmap.

External pull requests only proposing small or trivial changes will be converted to an issue and closed.

External contributions will require the signature of a Contributor License Agreement (CLA).
You can contact us using the following email to request a copy of the CLA: [lattice@lux.network](mailto:lattice@lux.network).

## Vulnerability Reports
See [Report a Vulnerability](SECURITY.md#report-a-vulnerability).

## Bug Reports

Lattice welcomes bug/regression reports of any kind that conform to the preset template, which is
automatically generated upon creation of a new empty issue. Nonconformity will result in the issue
being closed without acknowledgement.


## License

Lattice is licensed under the Apache 2.0 License. See [LICENSE](https://github.com/luxfi/lattice/blob/master/LICENSE).

## Contact

Before contacting us directly, please make sure that your request cannot be handled through an issue.

If you want to contribute to Lattice or report a security issue, you have a feature proposal or request, or you simply want to contact us directly, please do so using the following email: [lattice@lux.network](mailto:lattice@lux.network).

## Citing

Please use the following BibTex entry for citing Lattice:

    @misc{lattice,
	    title = {Lattice v6},
	    howpublished = {Online: \url{https://github.com/luxfi/lattice}},
	    month = Aug,
	    year = 2024,
	    note = {EPFL-LDS, Tune Insight SA}
    }


The Lattice logo is a lattice-based version of the original Golang mascot by [Renee
French](http://reneefrench.blogspot.com/).
