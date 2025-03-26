# ============== Implementing Matlab imresize ================================================================
# Tien C. Bau, transcoded from Matlab.
# ============================================================================================================
import numpy as np


def cubic(x):
    # See Keys, "Cubic Convolution Interpolation for Digital Image
    # Processing," IEEE Transactions on Acoustics, Speech, and Signal
    # Processing, Vol. ASSP-29, No. 6, December 1981, p. 1155.

    absx = np.absolute(x);
    absx2 = np.power(absx, 2);
    absx3 = np.power(absx, 3);

    f = (1.5 * absx3 - 2.5 * absx2 + 1) * (absx <= 1).astype(np.float32) + (
                -0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * ((1 < absx) & (absx <= 2)).astype(np.float32)
    return f


def box(x):
    f = ((-0.5 <= x) & (x < 0.5)).astype(np.float32)
    return f


def triangle(x):
    f = (x + 1) * ((-1 <= x) & (x < 0)).astype(np.float32) + (1 - x) * ((0 <= x) & (x <= 1)).astype(np.float32);
    return f


def lanczos2(x):
    # See Graphics Gems, Andrew S. Glasser (ed), Morgan Kaufman, 1990,
    # pp. 156-157.

    f = (np.sin(np.pi * x) * np.sin(np.pi * x / 2) + np.finfo(np.float32).eps) / (
                (np.pi ** 2 * np.power(x, 2) / 2) + np.finfo(np.float32).eps);
    f = f * (np.absolute(x) < 2).astype(np.float32);
    return f


def lanczos3(x):
    # See Graphics Gems, Andrew S. Glasser (ed), Morgan Kaufman, 1990,
    # pp. 157-158.

    f = (np.sin(np.pi * x) * np.sin(np.pi * x / 3) + np.finfo(np.float32).eps) / (
                (np.pi ** 2 * np.power(x, 2) / 3) + np.finfo(np.float32).eps);
    f = f * (np.absolute(x) < 3).astype(np.float32);
    return f


def matlab_imresize_contributions(in_length, out_length, scale, kernel, kernel_width, antialiasing,
                                  align_nearest=False):
    if (scale < 1) and (antialiasing):
        # Use a modified kernel to simultaneously interpolate and
        # antialias.
        h = lambda x: scale * kernel(scale * x)
        kernel_width = kernel_width / scale
    else:
        # No antialiasing; use unmodified kernel.
        h = kernel
    # Output-space coordinates.
    x = np.float32(np.arange(out_length)).reshape(int(out_length), 1) + 1

    # Input-space coordinates. Calculate the inverse mapping such that 0.5
    # in output space maps to 0.5 in input space, and 0.5+scale in output
    # space maps to 1.5 in input space.
    if (align_nearest):
        u = x / scale + 1. * (1. - 1. / scale)
    else:
        u = x / scale + 0.5 * (1. - 1. / scale)

    # What is the left-most pixel that can be involved in the computation?
    left = np.floor(u - kernel_width / 2.)

    # What is the maximum number of pixels that can be involved in the
    # computation?  Note: it's OK to use an extra pixel here; if the
    # corresponding weights are all zero, it will be eliminated at the end
    # of this function.
    P = np.ceil(kernel_width) + 2.

    # The indices of the input pixels involved in computing the k-th output
    # pixel are in row k of the indices matrix.
    # indices = bsxfun(@plus, left, 0:P-1);
    indices = np.repeat(np.arange(P).reshape(1, int(P)).astype(np.float32), out_length, axis=0) + np.repeat(left, P,
                                                                                                            axis=1)

    # The weights used to compute the k-th output pixel are in row k of the
    # weights matrix.
    # weights = h(bsxfun(@minus, u, indices));
    weights = h(np.repeat(u, P, axis=1) - indices)

    # Normalize the weights matrix so that each row sums to 1.
    # weights = bsxfun(@rdivide, weights, sum(weights, 2));
    weights = weights / np.repeat(np.sum(weights, axis=1).reshape(weights.shape[0], 1), P, axis=1)

    # Clamp out-of-range indices; has the effect of replicating end-points.
    indices = np.minimum(np.maximum(0, indices - 1), in_length - 1)

    # If a column in weights is all zero, get rid of it.
    # kill = find(~any(weights, 1));
    # if ~isempty(kill)
    #    weights(:,kill) = [];
    #    indices(:,kill) = [];
    # end
    return indices, np.float32(weights)


def getMethodInfo(method):
    method_func = {'nearest': box,
                   'bilinear': triangle,
                   'bicubic': cubic,
                   'box': box,
                   'triangle': triangle,
                   'cubic': cubic,
                   'lanczos2': lanczos2,
                   'lanczos3': lanczos3}

    method_kern = {'nearest': 1.0,
                   'bilinear': 2.0,
                   'bicubic': 4.0,
                   'box': 1.0,
                   'triangle': 2.0,
                   'cubic': 4.0,
                   'lanczos2': 4.0,
                   'lanczos3': 6.0}

    return method_func[method], method_kern[method]


def resizeAlongDim0(im, dim, weights, indices):
    out_length = weights.shape[0]
    # block_size = 15;
    out = np.zeros([indices.shape[0], im.shape[1], im.shape[2]], np.float32)
    for c in range(im.shape[2]):
        block = im[:, :, c].astype(np.float32)
        mac_s = np.zeros([indices.shape[0], im.shape[1]], np.float32)
        for i in range(indices.shape[1]):
            # mac_i = np.repeat(indices[:,i].reshape(indices.shape[0],1),im.shape[1],axis=1)
            mac_i = indices[:, i].astype(int)
            mac_w = np.repeat(weights[:, i].reshape(weights.shape[0], 1), im.shape[1], axis=1)
            mac_s = mac_s + block[mac_i, :] * mac_w
        out[:, :, c] = mac_s

        # for p in range(0,out_length,block_size):
        #    pp = range(p,(min(p + block_size - 1, out_length)));
        #    block = im[pp,:,c];
        #    block = resizeColumns(block, weights, indices);
        #    out[mm, :, pp] = ipermute(block, [2,1,3]);
    return out


def resizeAlongDim1(im, dim, weights, indices):
    out_length = weights.shape[1];
    # block_size = 15;
    out = np.zeros([im.shape[0], indices.shape[0], im.shape[2]], np.float32)
    for c in range(im.shape[2]):
        block = im[:, :, c].astype(np.float32)
        mac_s = np.zeros([im.shape[0], indices.shape[0]], np.float32)
        for i in range(indices.shape[1]):
            # mac_i = np.repeat(indices[:,i].reshape(indices.shape[0],1),im.shape[1],axis=1)
            mac_i = indices[:, i].astype(int)
            mac_w = np.repeat(weights[:, i].reshape(1, weights.shape[0]), im.shape[0], axis=0)
            mac_s = mac_s + block[:, mac_i] * mac_w
        out[:, :, c] = mac_s

        # for p in range(0,out_length,block_size):
        #    pp = range(p,(min(p + block_size - 1, out_length)));
        #    block = im[pp,:,c];
        #    block = resizeColumns(block, weights, indices);
        #    out[mm, :, pp] = ipermute(block, [2,1,3]);
    return out


def matlab_imresize(A, ratio=None, interp='bilinear', antialiasing=True, align_nearest=False,
                    limit_range=False, maxcode=None, mincode=None, outsize=None):
    '''
    ratio = [scaley,scalex]
    outsize=[sizey,sizex]
    interp: interpolation methods, 'bilinear', 'bicubic', 'nearest', 'box','triangle' (same as bilinear), 'cubic', 'lanczos2', 'lanczos3'
    antialiasing:  to match the one in matlab.
    align_nearest: when other scaling method needs to align with 'nearest', set this to true.
    limit_range:   to limit the range of the final output, watch out matlab doesn't have this option. When input has negative value, the output should be allow to have negative as well.

    Previously the DoG feature may have bug when scaling the input in float mode. To match the bug, use limit_range=True, maxcode=1, mincode=0
    '''
    kernel, kernel_width = getMethodInfo(interp)
    weights = [None] * 2
    indices = [None] * 2
    if (ratio is not None):
        ratio = [ratio, ratio] if not isinstance(ratio, (list, tuple)) else ratio
        if (ratio[0] == 1.0 and ratio[1] == 1.0):
            return A
        outsize = np.ceil([np.float32(A.shape[0]) * ratio[0], np.float32(A.shape[1]) * ratio[1]])
    elif (outsize is not None):
        outsize = [outsize, outsize] if not isinstance(outsize, (list, tuple)) else outsize
        ratio = [np.float32(outsize[0]) / np.float32(A.shape[0]), np.float32(outsize[1]) / np.float32(A.shape[1])]
        if (ratio[0] == 1.0 and ratio[1] == 1.0):
            return A

    for k in range(2):
        indices[k], weights[k] = matlab_imresize_contributions(
            in_length=A.shape[k],
            out_length=outsize[k],
            scale=ratio[k],
            kernel=kernel,
            kernel_width=kernel_width,
            antialiasing=antialiasing,
            align_nearest=align_nearest)
    B = A
    B = resizeAlongDim0(B, 0, weights[0], indices[0])
    B = resizeAlongDim1(B, 1, weights[1], indices[1])
    if (limit_range):
        if (maxcode is None):
            if (A.dtype == 'uint16'):
                maxcode = 65535.
            elif (A.dtype == 'uint8'):
                maxcode = 255.
            else:
                maxcode = np.inf
        if (mincode is None):
            if (A.dtype == 'uint16'):
                mincode = 0.
            elif (A.dtype == 'uint8'):
                mincode = 0.
            else:
                mincode = -np.inf
        B = np.minimum(np.maximum(mincode, B), maxcode).astype(A.dtype)
    return B


