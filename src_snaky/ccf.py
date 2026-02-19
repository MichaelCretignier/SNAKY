import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline
from tqdm import tqdm


# PLutôt se baser sur le deuxième benchmark pour tester ! :ok
# wave          => Array des infos de la longueur d'onde (1xW)
# spec1         => vecteur de (NxW) valeurs en float
# spec2         => masque binaire (1xW) 1 masque qu'on coulisse
# extended      => nombre d'éléments à ajouter avant / après
# rv_range      => nombre de coulissements
# oversampling  => Permet d'affiner la donnée
def ccf(
    wavelenghts: NDArray[np.float64],
    spectrums: NDArray[np.float64],
    mask: NDArray[np.bool_],
    shift: int = 1500,
    shift_no: int = 45,
    oversampling: int = 10,
    spectrums_uncertainty: NDArray[np.float64] | None = None,
):
    "CCF for a equidistant grid in log wavelength spec1 = spectrum, spec2 =  binary mask"
    dwave: np.float64 = np.median(np.diff(wavelenghts))

    if spectrums_uncertainty is None:
        spectrums_uncertainty = np.zeros(np.shape(spectrums))

    if len(np.shape(spectrums)) == 1:
        spectrums = spectrums[:, np.newaxis].T

    if len(np.shape(spectrums_uncertainty)) == 1:
        spectrums_uncertainty = spectrums_uncertainty[:, np.newaxis].T

    spectrums = np.hstack(
        [np.ones((len(spectrums), shift)), spectrums, np.ones((len(spectrums), shift))]
    )
    mask_shift_vector: NDArray[np.bool_] = np.zeros(shift, dtype=bool)
    mask = np.hstack([mask_shift_vector, mask, mask_shift_vector])

    spectrums_uncertainty_shift = np.zeros((len(spectrums_uncertainty), shift))
    spectrums_uncertainty = np.hstack(
        [
            spectrums_uncertainty_shift,
            spectrums_uncertainty,
            spectrums_uncertainty_shift,
        ]
    )
    wavelenghts = np.hstack(
        [
            np.arange(-shift * dwave + wavelenghts.min(), wavelenghts.min(), dwave),
            wavelenghts,
            np.arange(
                wavelenghts.max() + dwave,
                (shift + 1) * dwave + wavelenghts.max(),
                dwave,
            ),
        ]
    )

    shifts: NDArray[np.float64] = np.linspace(0, dwave, oversampling + 1)[:-1]
    sum_spec: int = np.nansum(mask, dtype=int)
    convolutions = []
    convolutions_uncertainty = []

    shift_save = []
    rv_max = int(np.log10((shift_no / 299.792e3) + 1) / dwave)
    for shift_el in tqdm(shifts):
        new_spec = CubicSpline(wavelenghts + shift_el, mask, extrapolate=True)(
            wavelenghts
        )
        for k in np.arange(-rv_max, rv_max + 1, 1):
            new_spec2 = np.hstack([new_spec[-k:], new_spec[:-k]])
            convolutions.append(np.nansum(new_spec2 * spectrums, axis=1) / sum_spec)
            convolutions_uncertainty.append(
                np.sqrt(np.abs(np.nansum(new_spec2 * spectrums_uncertainty**2, axis=1)))
                / sum_spec
            )
            shift_save.append(shift_el + k * dwave)
    shift_save = np.array(shift_save)
    sorting = np.argsort(shift_save)

    return (
        (299.792e6 * 10 ** shift_save[sorting])
        - 299.792e6,  # données vitesse de la lumière
        np.array(convolutions)[sorting],  # matrice de de CCF
        np.array(convolutions_uncertainty)[
            sorting
        ],  # erreurs que tu as => osef ( même taille que CCF)
    )
