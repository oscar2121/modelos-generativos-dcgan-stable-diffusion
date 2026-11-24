"""
Implementación Modular de una Red Generativa Adversaria Convolucional Profunda (DCGAN)
para la Generación Sintética de Rostros Humanos (64x64 RGB).
Autor: Oscar - Programa Académico en Inteligencia (Noviembre de 2026)
Referencias: Goodfellow et al. (2014); Radford et al. (2015).
"""

import torch
import torch.nn as nn


def weights_init(module: nn.Module) -> None:
    """Inicialización gaussiana N(0, 0.02) recomendada por Radford et al. (2015)."""
    classname = module.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)


class DCGANGenerator(nn.Module):
    """
    Componente Generator (G): Mapea un vector del espacio latente z ~ N(0, I_100)
    hacia el espacio de imágenes RGB [-1, 1] de resolución 64x64x3.
    """

    def __init__(self, nz: int = 100, ngf: int = 64, nc: int = 3) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.main(z)


class DCGANDiscriminator(nn.Module):
    """
    Componente Discriminator (D): Clasificador binario convolucional que estima
    la probabilidad D(x) de que una imagen 64x64x3 sea un rostro real.
    """

    def __init__(self, nc: int = 3, ndf: int = 64) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x).view(-1)
