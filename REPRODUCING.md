This [Code Ocean](https://codeocean.com) compute capsule will allow you to reproduce the results published by the author on your local machine<sup>1</sup>. Follow the instructions below, or consult [our knowledge base](https://help.codeocean.com/user-manual/sharing-and-finding-published-capsules/exporting-capsules-and-reproducing-results-on-your-local-machine) for more information. Don't hesitate to reach out via live chat or [email](mailto:support@codeocean.com) if you have any questions.

<sup>1</sup> You may need access to additional hardware and/or software licenses.

# Prerequisites

- [Docker Community Edition (CE)](https://www.docker.com/community-edition)
- MATLAB/MOSEK/Stata licenses where applicable

# Instructions

## The computational environment (Docker image)

This capsule has been published and its environment has been archived and made available on Code Ocean's Docker registry:
`registry.codeocean.com/published/a0f2899b-3f88-493a-b1a7-f379435d004c:v1`

## Running the capsule to reproduce the results

In your terminal, navigate to the folder where you've extracted the capsule and execute the following command, adjusting parameters as needed:
```shell
docker run --rm \
  --workdir /code \
  --volume "$PWD/data":/data \
  --volume "$PWD/code":/code \
  --volume "$PWD/results":/results \
  registry.codeocean.com/published/a0f2899b-3f88-493a-b1a7-f379435d004c:v1 ./run
```
